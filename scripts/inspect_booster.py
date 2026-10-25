"""Read-only full processed-dataset audit; writes only to --output-dir."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from src.common import read_json, write_json, protect_dataset, digest_file
from src.datasets.booster import BoosterDataset, MATERIAL_CLASSES, MATERIAL_SOURCE


def audit(root, raw_root=None, preprocess_code_root=None):
    root = Path(root)
    reports, errors, warnings = {}, [], []
    file_types, missing_files = Counter(), []
    scene_splits = {}
    for index in sorted(root.glob("*/index.json")):
        split = index.parent.name
        ds = BoosterDataset(root, split)
        items, counts, subsets = [], Counter(), Counter()
        indexed_dirs = {str(e["dir"]).replace("\\", "/") for e in ds.samples}
        discovered = {p.parent.relative_to(ds.root).as_posix() for p in ds.root.rglob("meta.json")}
        if discovered != indexed_dirs:
            errors.append(dict(split=split, error="Index/disk mismatch",
                               unindexed=sorted(discovered-indexed_dirs),
                               missing_dirs=sorted(indexed_dirs-discovered)))
        for i in range(len(ds)):
            sid = ds.sample_id(i)
            try:
                paths = ds.paths(i)
                for key, path in paths.items():
                    if not path.is_file():
                        missing_files.append(dict(split=split, sample_id=sid, kind=key))
                    else:
                        file_types[f"{key}:{path.suffix}"] += 1
                s = ds[i]
                d, m = s["depth_gt"], s["meta"]
                finite_positive = np.isfinite(d) & (d > 0)
                t = m["transform"]
                matrix = np.array([[t["scale_x"], 0, -t["crop_x"]],
                                   [0, t["scale_y"], -t["crop_y"]], [0, 0, 1]])
                if not np.allclose(s["updated_K"], matrix @ s["original_K"], atol=1e-5):
                    raise ValueError("Saved K differs from T_crop @ S_resize @ original_K")
                oh, ow = s["original_size"]
                rh, rw = t["resized_size"]
                if not np.allclose([rw/ow, rh/oh], [t["scale_x"], t["scale_y"]]):
                    raise ValueError("Incorrect transform scale")
                if not np.array_equal(s["saved_valid_mask"], finite_positive):
                    raise ValueError("Saved valid_mask differs from finite positive depth")
                material = s["material_mask"]
                mc = Counter() if material is None else Counter(map(int, material.ravel()))
                counts.update(mc)
                subsets[sid.split('/')[0]] += 1
                scene_splits.setdefault(s["scene_id"], set()).add(split)
                vals = d[finite_positive]
                items.append(dict(sample_id=sid, shape=list(d.shape), dtype=str(d.dtype),
                                  original_size=list(s["original_size"]),
                                  depth_unit=m["depth_unit"], valid_pixels=int(vals.size),
                                  valid_ratio=float(finite_positive.mean()), zero=int((d == 0).sum()),
                                  negative=int((d < 0).sum()), nan=int(np.isnan(d).sum()),
                                  inf=int(np.isinf(d).sum()),
                                  depth_min_m=float(vals.min()) if vals.size else None,
                                  depth_max_m=float(vals.max()) if vals.size else None,
                                  material_counts=dict(mc),
                                  consistency_ratio=float(s["consistency_mask"].mean()),
                                  intrinsics_transform_passed=True))
                if material is None:
                    warnings.append(dict(sample_id=sid, warning="Material mask unavailable"))
                for warning in m.get("raw_meta", {}).get("calibration_warnings", []):
                    warnings.append(dict(sample_id=sid, warning=warning))
            except Exception as exc:
                errors.append(dict(split=split, sample_id=sid, error=f"{type(exc).__name__}: {exc}"))
        reports[split] = dict(index_samples=len(ds), readable_samples=len(items),
                              subsets=dict(subsets), material_pixel_counts=dict(counts), samples=items)
    if not reports:
        errors.append(dict(error="No split/index.json found; dataset structure unverified"))
    overlap = {s: sorted(v) for s, v in scene_splits.items() if len(v) > 1}
    if overlap:
        errors.append(dict(error="Scene leakage across splits", scenes=overlap))
    raw_split_check = None
    if raw_root:
        raw_root = Path(raw_root)
        raw_scenes = {}
        for split in ["train", "test", "val"]:
            scene_names = set()
            for subset in ["balanced", "unbalanced"]:
                directory = raw_root / split / subset
                if directory.is_dir():
                    scene_names.update(p.name for p in directory.iterdir() if p.is_dir())
            if scene_names:
                raw_scenes[split] = sorted(scene_names)
        overlaps = {f"{a}:{b}": sorted(set(raw_scenes[a]) & set(raw_scenes[b]))
                    for a in raw_scenes for b in raw_scenes if a < b}
        raw_split_check = dict(scene_names=raw_scenes, scene_name_overlap=overlaps,
                               interpretation="Name overlap is a leakage warning, not proof of identical physical scenes/images")
        if any(overlaps.values()):
            warnings.append(dict(warning="Raw split scene-name overlap requires review before fine-tuning", overlaps=overlaps))
    preprocess_source = None
    if preprocess_code_root:
        source = Path(preprocess_code_root)
        preprocess_source = {name: digest_file(source / name) for name in
                             ["README.md", "depth_pipeline/preprocess.py", "depth_pipeline/dataloader.py"]}
    return dict(schema_version=1, splits=reports, errors=errors, warnings=warnings,
                file_types=dict(file_types), missing_files=missing_files,
                missing_file_count=len(missing_files),
                raw_split_check=raw_split_check, preprocess_source_sha256=preprocess_source,
                material_classes=MATERIAL_CLASSES, material_source=MATERIAL_SOURCE,
                region_protocol=dict(target=[2, 3], ordinary=[0], other=[0, 1],
                                     separate_transparent_mirror="unavailable in these masks"),
                valid_definition="saved valid_mask AND finite GT AND GT > 0",
                consistency_mask="Separate stereo consistency mask; excluded from default validity",
                scene_overlap_across_processed_splits=overlap,
                limitations=["Only processed indexed splits audited; absence of test GT is not a test evaluation",
                             "Metadata K transform verified, not physical camera calibration accuracy",
                             "Existing mtxL vs projection focal discrepancy retained; no GT recomputation"],
                tree=["<root>/train/index.json",
                      "<root>/train/{balanced,unbalanced}/<scene>/camera_00/",
                      "  rgb.png, depth_gt.npy, valid_mask.png, mask.png, material_mask.png, meta.json",
                      "<root>/_pointcloud/, _pred_gt/, train/_viz/ (derived outputs; ignored)"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("docs"))
    p.add_argument("--raw-root", type=Path, help="Optional existing raw root; inspect split names only, no reprocessing")
    p.add_argument("--preprocess-code-root", type=Path, help="Optional inspected source root for provenance hashes")
    args = p.parse_args()
    protect_dataset(args.data_root, args.output_dir)
    if args.raw_root:
        protect_dataset(args.raw_root, args.output_dir)
    report = audit(args.data_root, args.raw_root, args.preprocess_code_root)
    write_json(args.output_dir / "dataset_audit.json", report)
    lines = ["# Observed processed Booster structure", "", "```text", *report["tree"], "```", "",
             "This report is from actual file reads. The data root is intentionally omitted.", "",
             "GT is saved metric depth in metres, not disparity. Loader performs no resize or unit conversion.", "",
             "Each indexed subset/scene/camera ID pairs all six files through meta.json/files.", "",
             f"File counts by role/extension: {report['file_types']}",
             f"Missing files: {report['missing_file_count']}", ""]
    for split, r in report["splits"].items():
        lines.extend([f"- Split `{split}`: {r['index_samples']} indexed; {r['readable_samples']} readable.",
                      f"- Subsets: {r['subsets']}",
                      f"- Shapes/dtypes: {sorted({str((s['shape'], s['dtype'])) for s in r['samples']})}",
                      f"- Material pixel counts: {r['material_pixel_counts']}"])
        if r["samples"]:
            lines.append(f"- Depth range (m): {min(s['depth_min_m'] for s in r['samples'] if s['depth_min_m'] is not None):.9f} to "
                         f"{max(s['depth_max_m'] for s in r['samples'] if s['depth_max_m'] is not None):.9f}")
    lines.extend(["", f"Errors: {len(report['errors'])}; warnings: {len(report['warnings'])} (details in dataset_audit.json).", "",
                  "Valid pixels: " + report["valid_definition"], "", report["consistency_mask"], "",
                  "Material labels describe severity, not separate transparent and mirror classes:", ""])
    lines.extend(f"- {k}: {v}" for k, v in MATERIAL_CLASSES.items())
    lines.extend(["", f"Definition: [Booster paper]({MATERIAL_SOURCE}).",
                  "Project protocol: target = 2+3, ordinary = 0, additional other = 0+1; all four classes are also reported.",
                  "", "Only train is currently indexed. No validation/test split is invented. Balanced/unbalanced views",
                  "share scene identities and must stay together in any future train/validation/test partition.", "",
                  "Preprocessing source inspection: original 3008x4112 -> area RGB/depth resize 448x612 -> centre crop",
                  "(x=26,y=0) to 448x560. Depth uses valid-aware area averaging; material uses nearest neighbour.",
                  "K uses actual scale_x/scale_y and crop offsets. Original disparity was converted once using",
                  "fx(mtxL)*baseline_mm/disparity/1000 before resizing. Saved GT is reused unchanged.", "",
                  "Known calibration caveat: mtxL and proj_matL focal lengths differ in the existing preprocessing",
                  "(scene-dependent; see exact warnings in the JSON). These measurements are conditional on the existing",
                  "mtxL-based GT; this audit does not establish which calibration model is physically preferable.", "",
                  "Mask PNGs are stored as RGB with identical channels by the existing writer; the loader validates",
                  "channel equality before reducing to one channel. No class values are inferred from colour.", "",
                  *report["limitations"]])
    if report["raw_split_check"]:
        lines.extend(["", "Raw split name check (no depth conversion/reprocessing):",
                      str(report["raw_split_check"]["scene_name_overlap"]),
                      report["raw_split_check"]["interpretation"],
                      "Do not reuse overlapping scene names in separate fine-tuning splits without resolving this warning."])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "DATASET_STRUCTURE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Audited {sum(r['readable_samples'] for r in report['splits'].values())} samples; errors={len(report['errors'])}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
