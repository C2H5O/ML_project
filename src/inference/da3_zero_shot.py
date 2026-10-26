"""Official DA3 API wrapper with explicit precision and provenance."""
import importlib.metadata
import subprocess
from pathlib import Path
import types
import time

import numpy as np

DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_ID = "depth-anything/DA3METRIC-LARGE"
MODELS = (MODEL_ID,)
METRIC_PROTOCOL = "da3metric_focal_over_300_v1"


def metric_depth_from_canonical(depth, input_K, input_shape):
    """Official Metric-Large focal convention, at the actual network pixel grid.

    Pinned API does not apply this conversion to the standalone metric model.
    Resize-only preprocessing scales rows of K, including actual patch rounding.
    No GT, scene statistic, baseline or predicted camera participates.
    """
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(input_K, dtype=np.float64)
    if depth.ndim != 2 or len(input_shape) != 2 or min(input_shape) <= 0:
        raise ValueError("Expected native HxW depth and positive saved input dimensions")
    if (K.shape != (3, 3) or not np.all(np.isfinite(K)) or
            K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("Missing/unconfirmed calibrated pixel intrinsics")
    h, w = input_shape
    nh, nw = depth.shape
    native_K = np.diag([nw / w, nh / h, 1.0]) @ K
    focal = float((native_K[0, 0] + native_K[1, 1]) / 2)
    factor = focal / 300.0
    metric = depth.astype(np.float64) * factor
    if np.any(np.isfinite(metric) & (np.abs(metric) > np.finfo(np.float32).max)):
        raise ValueError("Metric depth float32 overflow")
    return metric.astype(np.float32), dict(
        protocol=METRIC_PROTOCOL, input_K=K.tolist(), native_K=native_K.tolist(),
        input_shape=[int(h), int(w)], native_shape=[int(nh), int(nw)],
        focal_native_px=focal, canonical_focal_px=300.0, multiplier=factor,
        formula="depth_m = canonical_output * mean(fx_native, fy_native) / 300",
        gt_used=False, applied_exactly_once=True)


def metric_flag(value):
    # Upstream AddictDict can create an empty Dict on getattr of a missing key.
    # Preserve that as unknown rather than pretending it is a metric flag.
    return int(value) if isinstance(value, (bool, int, float, np.number)) and value in (0, 1) else None


def resize_prediction(depth, shape):
    """Bilinear predictions only, never resample GT/masks.

    Invalid prediction neighbourhoods stay invalid instead of blending into valid
    values. This is independent of GT; border support uses PIL bilinear semantics.
    """
    from PIL import Image
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW prediction, got {depth.shape}")
    if depth.shape == tuple(shape):
        return depth.copy()
    valid = np.isfinite(depth) & (depth > 0)
    size = (int(shape[1]), int(shape[0]))
    values = np.asarray(Image.fromarray(np.where(valid, depth, 0)).resize(size, Image.Resampling.BILINEAR))
    support = np.asarray(Image.fromarray(valid.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    return np.where(support >= 1 - 1e-6, values, np.nan).astype(np.float32)


def precision_for_device(torch, device, requested):
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"Unknown precision: {requested}")
    if device.type == "cpu":
        if requested not in {"auto", "fp32"}:
            raise ValueError("CPU supports fp32 in this baseline")
        return "fp32"
    with torch.cuda.device(device):
        bf16 = torch.cuda.is_bf16_supported()
    if requested == "bf16" and not bf16:
        raise ValueError("Selected GPU does not support bf16; use --precision fp16")
    return ("bf16" if bf16 else "fp16") if requested == "auto" else requested


class DA3Predictor:
    def __init__(self, model_id=MODELS[0], checkpoint=None, device="auto", precision="auto",
                 revision=None, require_gpu="RTX 4090D"):
        import torch
        import sys
        from src.runtime import runtime_issues
        issues = runtime_issues(sys.version_info, torch.__version__)
        if issues:
            raise RuntimeError("Existing environment is incompatible; do not auto-install/upgrade: " + "; ".join(issues))
        try:
            import depth_anything_3
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise RuntimeError("DA3 import failed in the existing environment; inspect dependencies without auto-installing: " + str(exc)) from exc
        if model_id not in MODELS:
            raise ValueError(f"Supported official model IDs: {MODELS}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only cpu/cuda devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        from src.runtime import require_gpu_name
        require_gpu_name(torch, self.device, require_gpu)
        self.precision = precision_for_device(torch, self.device, precision)
        package_dir = Path(next(iter(depth_anything_3.__path__))).resolve()
        try:
            code_commit = subprocess.check_output(["git", "-C", str(package_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            changed = subprocess.check_output(["git", "-C", str(package_dir), "diff", "HEAD", "--name-only"], text=True).splitlines()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("Cannot establish official DA3 source provenance; no weights were loaded") from exc
        if code_commit != DA3_COMMIT:
            raise RuntimeError(f"Unsupported DA3 source commit {code_commit}; expected {DA3_COMMIT}. Do not change the existing environment automatically.")
        if any(path != "src/depth_anything_3/utils/export/__init__.py" for path in changed):
            raise RuntimeError("Official DA3 source has unreviewed computation changes; inspect them before metric inference")
        source = model_id
        checkpoint_hashes = None
        if checkpoint:
            from src.common import digest_file
            source_path = Path(checkpoint)
            if not source_path.is_dir() or not all((source_path / f).is_file() for f in ("config.json", "model.safetensors")):
                raise FileNotFoundError("--checkpoint must be a local HF directory with config.json and model.safetensors")
            import json
            config = json.loads((source_path / "config.json").read_text())
            expected = model_id.rsplit('/', 1)[1].lower()
            if config.get("model_name") != expected:
                raise ValueError(f"Local checkpoint architecture {config.get('model_name')} disagrees with {model_id}")
            source = str(source_path.resolve())
            checkpoint_hashes = {f: digest_file(source_path / f) for f in ("config.json", "model.safetensors")}
        try:
            # Resolve a moving Hub ref once; subsequent loading uses its immutable SHA.
            resolved_revision = None
            if not checkpoint:
                from huggingface_hub import HfApi
                resolved_revision = HfApi().model_info(model_id, revision=revision or "main").sha
            self.model = DepthAnything3.from_pretrained(source, **(
                {"revision": resolved_revision} if not checkpoint else {"local_files_only": True}))
        except Exception as exc:
            raise RuntimeError(f"Cannot load official {model_id}; check checkpoint, network, HF cache and revision: {exc}") from exc
        expected = model_id.rsplit('/', 1)[1].lower()
        if self.model.model_name != expected:
            raise ValueError("Loaded model architecture disagrees with requested ID")
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        selected_precision = self.precision

        # Official forward hardcodes AMP. Instance-local replacement duplicates
        # only its wrapper, preserving all network arguments and output logic.
        # This avoids an outer autocast being overridden by the official wrapper.
        def controlled_forward(instance, image, extrinsics=None, intrinsics=None,
                               export_feat_layers=None, infer_gs=False, use_ray_pose=False,
                               ref_view_strategy="saddle_balanced"):
            dtype = torch.bfloat16 if selected_precision == "bf16" else torch.float16
            with torch.inference_mode(), torch.autocast(image.device.type, dtype=dtype,
                                                       enabled=selected_precision != "fp32"):
                return instance.model(image, extrinsics, intrinsics, export_feat_layers,
                                      infer_gs, use_ray_pose, ref_view_strategy)

        self.model.forward = types.MethodType(controlled_forward, self.model)
        self.metadata = dict(model_id=model_id, architecture=self.model.model_name,
                             checkpoint_kind="local_hf_directory" if checkpoint else "huggingface",
                             checkpoint_hashes=checkpoint_hashes, hf_revision=resolved_revision,
                             da3_code_commit=code_commit, precision=self.precision, device=str(self.device),
                             da3_modified_files=changed,
                             gpu_name=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                             required_gpu=require_gpu,
                             precision_adapter="instance-local official forward AMP policy only",
                             python=__import__('platform').python_version(),
                             torch=torch.__version__, torch_cuda=torch.version.cuda,
                             runtime_versions={p: importlib.metadata.version(p) for p in
                                               ["numpy", "pillow", "huggingface-hub", "safetensors"]},
                             depth_semantics="raw_metric_metres", metric_protocol=METRIC_PROTOCOL,
                             gt_alignment=False)

    def predict(self, rgb, input_K, process_res=560, process_res_method="upper_bound_resize"):
        native, timing = self.predict_canonical(rgb, process_res, process_res_method)
        metric, calibration = metric_depth_from_canonical(native, input_K, rgb.shape[:2])
        return native, metric, dict(**timing, metric_conversion=calibration)

    def predict_canonical(self, rgb, process_res=560, process_res_method="upper_bound_resize"):
        """RGB-only official output, with no assumption about calibrated scale."""
        import torch
        if process_res_method not in {"upper_bound_resize", "lower_bound_resize"}:
            raise ValueError("Crop strategies are disallowed: output must cover the entire RGB/GT field of view")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB HWC; DA3 performs its own preprocessing")
        gpu = self.device.type == "cuda"
        if gpu:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                prediction = self.model.inference(image=[rgb], process_res=process_res,
                                                   process_res_method=process_res_method,
                                                   infer_gs=False, export_dir=None)
            if gpu:
                torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError("CUDA OOM: release other GPU jobs or lower --process-res explicitly in a new output directory; model remains DA3METRIC-LARGE") from exc
        seconds = time.perf_counter() - start
        native = np.asarray(prediction.depth)
        if native.ndim != 3 or native.shape[0] != 1:
            raise ValueError(f"Unexpected official Prediction.depth shape: {native.shape}")
        native = native[0].astype(np.float32)
        if not (np.isfinite(native) & (native > 0)).any():
            raise ValueError("Model returned no finite positive depth")
        if metric_flag(prediction.is_metric) == 1:
            raise ValueError("Unexpected already-metric upstream output; refusing possible double conversion")
        processed_images = np.asarray(prediction.processed_images)
        if processed_images.shape != (1, *native.shape, 3):
            raise ValueError("Official processed image grid and depth grid disagree")
        return native, dict(seconds=seconds, native_shape=list(native.shape),
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if gpu else None,
                            prediction_is_metric=metric_flag(prediction.is_metric))
