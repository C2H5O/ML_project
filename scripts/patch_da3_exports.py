"""Make optional official exporters lazy; fail closed on source drift.

No model, preprocessing, weights or output code is changed. The upstream
dispatcher and each export call are preserved. GS/colmap packages are needed
only if those optional exporters are actually invoked (this baseline never does).
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
TARGET = "src/depth_anything_3/utils/export/__init__.py"
IMPORTS = {
    "glb": "from .glb import export_to_glb",
    "mini_npz": "from .npz import export_to_mini_npz, export_to_npz",
    "npz": "from .npz import export_to_mini_npz, export_to_npz",
    "feat_vis": "from .feat_vis import export_to_feat_vis",
    "depth_vis": "from .depth_vis import export_to_depth_vis",
    "gs_ply": "from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video",
    "gs_video": "from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video",
    "colmap": "from .colmap import export_to_colmap",
}


def patched(original):
    value = original
    for statement in set(IMPORTS.values()):
        if original.count(statement + "\n") != 1:
            raise ValueError("Unexpected upstream import layout")
        value = value.replace(statement + "\n", "")
    for fmt, statement in IMPORTS.items():
        marker = f'export_format == "{fmt}":\n'
        if value.count(marker) != 1:
            raise ValueError("Unexpected upstream dispatcher layout")
        value = value.replace(marker, marker + f"        {statement}\n")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError(f"DA3 commit must be {COMMIT}; got {head}")
    original = subprocess.check_output(["git", "-C", str(args.source), "show", f"{COMMIT}:{TARGET}"], text=True)
    expected = patched(original)
    path = args.source / TARGET
    current = path.read_text(encoding="utf-8")
    changed = subprocess.check_output(["git", "-C", str(args.source), "diff", "--name-only"], text=True).splitlines()
    if any(name != TARGET for name in changed):
        raise ValueError("Unrelated source modifications in DA3 checkout")
    if args.verify:
        if current != expected:
            raise ValueError("Installed patch is not the exact expected lazy-export patch")
    elif current == original:
        path.write_text(expected, encoding="utf-8", newline="\n")
    elif current != expected:
        raise ValueError("Refusing to overwrite unknown source edits")
    print("Verified lazy exporter patch sha256=" + hashlib.sha256(expected.encode()).hexdigest())


if __name__ == "__main__":
    main()
