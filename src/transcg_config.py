"""Optional YAML configuration for TransCG commands; explicit CLI wins."""
import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = {"data_root", "checkpoint", "output_dir", "intrinsics", "calibration_root",
               "input_dir", "calibrated_output_dir"}
STRING_FIELDS = PATH_FIELDS | {"sampling", "device", "precision"}
INT_FIELDS = {"scene_start", "scene_end", "camera", "num_samples", "seed", "process_res"}
FLOAT_FIELDS = {"depth_max", "error_max"}
BOOL_FIELDS = {"resume", "zoom_regions"}
FIELDS = STRING_FIELDS | INT_FIELDS | FLOAT_FIELDS | {"intrinsics_shape"} | BOOL_FIELDS
NULLABLE = {"intrinsics", "intrinsics_shape", "calibration_root"}


def load_config(path):
    # PyYAML is already a dependency of the project's OmegaConf/DA3 stack.
    # Import only for --config so existing explicit CLI usage is unchanged.
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("TransCG config must be a flat YAML mapping")
    unknown = set(config) - FIELDS
    if unknown:
        raise ValueError(f"Unknown TransCG config keys: {sorted(map(str, unknown))}")
    for key, value in config.items():
        if value is None and key in NULLABLE:
            continue
        if key in STRING_FIELDS and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string")
        if key in INT_FIELDS and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        if key in FLOAT_FIELDS and (type(value) not in {int, float} or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be a finite positive number")
        if key in BOOL_FIELDS and type(value) is not bool:
            raise ValueError(f"{key} must be YAML true or false, not a quoted string")
        if key == "intrinsics_shape" and (not isinstance(value, list) or len(value) != 2 or
                any(type(v) is not int or v <= 0 for v in value)):
            raise ValueError("intrinsics_shape must be null or [positive HEIGHT, positive WIDTH]")
        if key in {"scene_start", "scene_end", "num_samples", "process_res"} and value <= 0:
            raise ValueError(f"{key} must be positive")
        choices = {"camera": {1, 2}, "sampling": {"global", "per_scene"},
                   "precision": {"auto", "bf16", "fp16", "fp32"}}
        if key in choices and value not in choices[key]:
            raise ValueError(f"Invalid {key}: {value}")
        if key in PATH_FIELDS:
            # Resolve relative YAML paths against the repository, independent
            # of the caller's CWD. Preserve Linux absolute paths when inspecting
            # a server config on Windows.
            path_value = Path(value).expanduser()
            if not path_value.is_absolute() and not value.startswith("/"):
                path_value = PROJECT_ROOT / path_value
            config[key] = str(path_value)
    if not 1 <= config.get("scene_start", 1) <= config.get("scene_end", 10) <= 10:
        raise ValueError("scene_start/scene_end must define a range within scene1-10")
    return config


class TransCGArgumentParser(argparse.ArgumentParser):
    """Translate validated config values into defaults parsed by argparse.

    Reusing CLI types/choices avoids unvalidated set_defaults values. Config
    tokens precede explicit CLI tokens, so explicit CLI values take precedence.
    Inspection/rendering accept the same file and use their relevant fields.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", type=Path, help="TransCG YAML file; CLI overrides YAML; relative YAML paths use project root")

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        # --help must remain usable without opening a config or loading DA3.
        if "--help" in argv or "-h" in argv:
            return super().parse_args(argv, namespace)
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", type=Path)
        selected, _ = pre.parse_known_args(argv)
        tokens = []
        if selected.config is not None:
            try:
                config = load_config(selected.config)
            except (OSError, ValueError, ImportError) as exc:
                self.error(f"Cannot read TransCG config: {exc}")
            except Exception as exc:
                # Includes YAML syntax errors with the source line/column.
                self.error(f"Invalid TransCG YAML: {exc}")
            actions = {action.dest: action for action in self._actions}
            for key, value in config.items():
                if key not in actions or value is None:
                    continue
                option = actions[key].option_strings[0]
                if key in BOOL_FIELDS:
                    tokens.append(option if value else "--no-" + option[2:])
                else:
                    tokens.append(option)
                    if isinstance(value, list):
                        tokens.extend(str(v) for v in value)
                    else:
                        tokens.append(str(value))
        return super().parse_args(tokens + argv, namespace)
