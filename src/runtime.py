"""Read-only compatibility policy for an existing research environment."""
import re


def runtime_issues(python_version, torch_version):
    issues = []
    py = tuple(python_version[:3])
    if not (3, 9, 0) <= py <= (3, 10, 20):
        issues.append("Python must satisfy official >=3.9 and server <=3.10.20")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch_version)
    if not match or not (2, 0, 0) <= tuple(map(int, match.groups())) <= (2, 3, 1):
        issues.append("PyTorch must satisfy official >=2.0 and server <=2.3.1")
    return issues


def require_gpu_name(torch, device, required):
    if not required:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Requested server GPU {required} requires an available CUDA device")
    actual = torch.cuda.get_device_name(device)
    normalize = lambda value: re.sub(r"[^A-Z0-9]", "", value.upper())
    if normalize(required) not in normalize(actual):
        raise RuntimeError(f"Required GPU {required}; selected device is {actual}. No inference was run.")
