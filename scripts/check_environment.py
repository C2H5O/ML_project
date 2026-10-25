"""Non-mutating runtime diagnostics. --strict enforces target Linux versions."""
import _bootstrap  # noqa: F401
import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from src.common import write_json


def check(import_da3=False):
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
            report["dependencies"][distribution] = dict(version=importlib.metadata.version(distribution), imported=True)
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
        if torch.__version__.split('+')[0] != "2.3.1" or torch.version.cuda != "12.1":
            report["issues"].append("Target requires torch 2.3.1 + CUDA 12.1 runtime")
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
    if sys.version_info[:3] != (3, 10, 20):
        report["issues"].append("Target requires Python 3.10.20")
    if platform.system() != "Linux":
        report["issues"].append("Target Linux/GLIBC compatibility cannot be validated on this host")
    if report["dependencies"].get("torchvision", {}).get("version", "").split('+')[0] != "0.18.1":
        report["issues"].append("Target requires torchvision 0.18.1")
    if import_da3:
        try:
            import depth_anything_3
            from depth_anything_3.api import DepthAnything3
            source = Path(next(iter(depth_anything_3.__path__))).resolve()
            report["da3_code_commit"] = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            from src.inference.da3_zero_shot import DA3_COMMIT
            if report["da3_code_commit"] != DA3_COMMIT:
                report["issues"].append("DA3 source commit mismatch")
            report["da3_api_imported"] = DepthAnything3 is not None
        except Exception as exc:
            report["issues"].append("DA3 API import failed: " + str(exc))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--import-da3", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = check(args.import_da3)
    import json
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)
    return 1 if args.strict and report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
