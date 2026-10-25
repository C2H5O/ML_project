"""Read-only diagnostics for the existing vggtomast3r research environment."""
import _bootstrap  # noqa: F401
import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from src.common import write_json


def check(import_da3=False, required_gpu=None, device="cuda:0"):
    report = dict(python=sys.version, platform=platform.platform(),
                  glibc=platform.libc_ver(), dependencies={}, issues=[])
    try:
        report["gcc"] = subprocess.check_output(["gcc", "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        report["gcc"] = None
    required = {"numpy": "numpy", "Pillow": "PIL", "opencv-python": "cv2", "matplotlib": "matplotlib",
                "einops": "einops", "huggingface-hub": "huggingface_hub", "safetensors": "safetensors",
                "omegaconf": "omegaconf", "addict": "addict", "imageio": "imageio", "tqdm": "tqdm",
                "evo": "evo", "scipy": "scipy", "torchvision": "torchvision"}
    for distribution, module in required.items():
        try:
            imported = importlib.import_module(module)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = str(getattr(imported, "__version__", "unknown (module import passed)"))
            report["dependencies"][distribution] = dict(version=version, imported=True)
        except Exception as exc:
            report["dependencies"][distribution] = dict(imported=False, error=str(exc))
            report["issues"].append(f"Missing/broken runtime dependency: {distribution}")
    try:
        import torch
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), gpus=[])
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            with torch.cuda.device(i):
                bf16 = torch.cuda.is_bf16_supported()
                # Small actual CUDA allocation/operation, not only driver discovery.
                value = (torch.ones(1, device=f"cuda:{i}") + 1).item()
            report["gpus"].append(dict(index=i, name=props.name, total_memory_bytes=props.total_memory,
                                       bf16_supported=bf16, cuda_operation_passed=value == 2))
        from src.runtime import runtime_issues, require_gpu_name
        report["issues"].extend(runtime_issues(sys.version_info, torch.__version__))
        if required_gpu:
            try:
                require_gpu_name(torch, torch.device(device), required_gpu)
                report["required_gpu_passed"] = True
            except RuntimeError as exc:
                report["required_gpu_passed"] = False
                report["issues"].append(str(exc))
        if not torch.cuda.is_available():
            report["issues"].append("Target GPU unavailable")
    except Exception as exc:
        report["issues"].append("Torch/CUDA check failed: " + str(exc))
    try:
        import xformers
        import xformers.ops
        report["xformers"] = dict(available=True, version=xformers.__version__, abi="importable; kernel ABI not tested or used")
    except Exception as exc:
        report["xformers"] = dict(available=False, detail=str(exc), required=False)
    if platform.system() != "Linux":
        report["issues"].append("Target Linux/GLIBC compatibility cannot be validated on this host")
    if import_da3:
        try:
            import depth_anything_3
            from depth_anything_3.api import DepthAnything3
            source = Path(next(iter(depth_anything_3.__path__))).resolve()
            report["da3_code_commit"] = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            from src.inference.da3_zero_shot import DA3_COMMIT
            if report["da3_code_commit"] != DA3_COMMIT:
                report["issues"].append("DA3 source commit mismatch")
            changed = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True).splitlines()
            report["da3_modified_files"] = changed
            if any(name != "src/depth_anything_3/utils/export/__init__.py" for name in changed):
                report["issues"].append("Unreviewed computation changes in official DA3 source")
            report["da3_api_imported"] = DepthAnything3 is not None
        except Exception as exc:
            report["issues"].append("DA3 API import failed: " + str(exc))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--import-da3", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--require-gpu", help="Fail strict preflight unless the selected GPU matches")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    report = check(args.import_da3, args.require_gpu, args.device)
    import json
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)
    return 1 if args.strict and report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
