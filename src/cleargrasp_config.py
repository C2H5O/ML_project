"""Validated YAML configuration shared by every ClearGrasp command.

All dataset/checkpoint/output paths come from YAML, never environment variables
or path overrides on the command line. Relative paths use the repository root.
"""
import argparse
import math
from pathlib import Path, PureWindowsPath
import re
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'cleargrasp_zero_shot.yaml'
PATH_FIELDS = {'data_root', 'checkpoint', 'output_dir'}
BOOL_FIELDS = {'all', 'skip_evaluation', 'skip_visualization', 'offline'}
INT_FIELDS = {'max_samples', 'seed', 'process_res', 'pair_draws', 'ring_radius', 'boundary_margin'}
FLOAT_FIELDS = {'gt_gap_m', 'compression_ratio', 'depth_max', 'error_max'}
INT_FIELDS |= {'analysis_expected_samples', 'analysis_min_pairs', 'analysis_min_unique_pixels',
               'analysis_boundary_width', 'analysis_min_region_pixels'}
FLOAT_FIELDS |= {'analysis_baseline_rtol', 'analysis_baseline_atol'}
FIELDS = PATH_FIELDS | BOOL_FIELDS | INT_FIELDS | FLOAT_FIELDS | {'split', 'device', 'precision', 'model_id'}


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError('ClearGrasp config must be a flat YAML mapping')
    unknown = set(config) - FIELDS
    if unknown:
        raise ValueError(f'Unknown ClearGrasp config keys: {sorted(map(str, unknown))}')
    for key, value in config.items():
        if key in PATH_FIELDS:
            if value is None and key == 'data_root':
                continue  # Explicit unresolved server path; execution rejects it.
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a nonempty path string')
            if re.search(r'\$[A-Za-z_{]|%[^%]+%', value):
                raise ValueError(f'{key}: environment-variable paths are forbidden; write the path in YAML')
            p = Path(value).expanduser()
            # Preserve foreign absolute paths for --print-config; never prefix CWD.
            if not p.is_absolute() and not value.startswith('/') and not PureWindowsPath(value).is_absolute():
                p = PROJECT_ROOT / p
            config[key] = p
        elif key in BOOL_FIELDS and type(value) is not bool:
            raise ValueError(f'{key} must be YAML true or false')
        elif key in INT_FIELDS and type(value) is not int:
            raise ValueError(f'{key} must be an integer')
        elif key in FLOAT_FIELDS and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f'{key} must be a finite number')
        elif key not in BOOL_FIELDS | INT_FIELDS | FLOAT_FIELDS and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'{key} must be a nonempty string')
    for key in ('max_samples', 'process_res', 'pair_draws', 'ring_radius'):
        if key in config and config[key] < 1:
            raise ValueError(f'{key} must be positive')
    for key in INT_FIELDS:
        if key.startswith('analysis_') and key in config and config[key] < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('analysis_baseline_rtol', 'analysis_baseline_atol'):
        if key in config and config[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if config.get('boundary_margin', 2) < 0:
        raise ValueError('boundary_margin must be nonnegative')
    if config.get('boundary_margin', 2) >= config.get('ring_radius', 30):
        raise ValueError('boundary_margin must be smaller than ring_radius')
    for key in ('gt_gap_m', 'depth_max', 'error_max'):
        if key in config and config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < config.get('compression_ratio', .25) < 1:
        raise ValueError('compression_ratio must be between zero and one')
    choices = {'split': {'real-test', 'real-val', 'synthetic-test', 'synthetic-val'},
               'precision': {'bf16', 'fp16', 'fp32'}, 'model_id': {'depth-anything/DA3METRIC-LARGE'}}
    for key, values in choices.items():
        if key in config and config[key] not in values:
            raise ValueError(f'Invalid {key}: {config[key]}')
    if config.get('offline', True) is not True:
        raise ValueError('ClearGrasp supports offline local-checkpoint inference only')
    return config


class ClearGraspArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, required_paths=('data_root', 'output_dir'), **kwargs):
        super().__init__(*args, **kwargs)
        self.required_paths = required_paths
        self.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                          help='YAML containing all paths; relative paths use the project root')
        self.add_argument('--print-config', action='store_true', help='Print resolved options without data/model access')

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        if '--help' in argv or '-h' in argv:
            return super().parse_args(argv, namespace)
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        selected, _ = pre.parse_known_args(argv)
        try:
            config = load_config(selected.config)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.error(f'Cannot load ClearGrasp config: {exc}')
        # Path fields intentionally have no CLI flags. Non-path flags can override YAML.
        self.set_defaults(**config)
        parsed = super().parse_args(argv, namespace)
        if '--max-samples' in argv or any(a.startswith('--max-samples=') for a in argv):
            parsed.all = False
        if not parsed.print_config:
            for key in self.required_paths:
                if key not in config or config[key] is None:
                    self.error(f'Set {key} in {parsed.config}; the server data path is not yet configured')
                value = getattr(parsed, key)
                if (sys.platform != 'win32' and PureWindowsPath(str(value)).drive or
                        sys.platform == 'win32' and str(config.get(key, '')).startswith('\\') and
                        not str(config.get(key, '')).startswith('\\\\')):
                    self.error(f'{key} belongs to another operating system; edit {parsed.config}')
        return parsed


def print_config(args):
    import json
    print(json.dumps(vars(args), indent=2, default=str))
