"""Flat YAML and repository-relative paths, isolated from legacy CLI parsing."""
import argparse
import math
from pathlib import Path, PureWindowsPath
import re
import sys
import yaml
from src.cleargrasp_config import PROJECT_ROOT, print_config

DEFAULTS = dict(data_root=None, transparent_output='results/cleargrasp_smoke',
    output_dir='results/cleargrasp_paired', checkpoint='checkpoints/DA3METRIC-LARGE',
    registration_review=None, expected_samples=20, process_res=560, device='cuda:0',
    precision='bf16', offline=True, seed=42, pair_draws=20000, gt_gap_m=.02,
    ring_radius=30, boundary_margin=2, compression_ratio=.25, erosion_margin=2,
    bootstrap_draws=10000, min_pairs=30, min_unique_pixels=10, depth_max=2., error_max=.5, difference_max=.5,
    log_difference_max=1., figure_dpi=120)
PATHS = ('data_root', 'transparent_output', 'output_dir', 'checkpoint', 'registration_review')


def parser(description=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/cleargrasp_paired.yaml')
    p.add_argument('--print-config', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse verified outputs; never overwrite predictions')
    return p


def parse_args(argv=None):
    p = parser(__doc__)
    a = p.parse_args(argv)
    raw = yaml.safe_load(a.config.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS):
        p.error('Expected flat paired YAML with known keys only')
    values = dict(DEFAULTS, **raw)
    for key, value in values.items():
        default = DEFAULTS[key]
        if key in PATHS:
            if value is None and key in ('data_root', 'registration_review'):
                continue
            if not isinstance(value, str) or not value.strip() or re.search(r'\$[A-Za-z_{]|%[^%]+%', value):
                p.error(f'{key}: write a literal path in YAML')
            path = Path(value).expanduser()
            if not path.is_absolute() and not value.startswith('/') and not PureWindowsPath(value).is_absolute():
                path = PROJECT_ROOT/path
            if not a.print_config and ((sys.platform != 'win32' and PureWindowsPath(value).drive) or
                    (sys.platform == 'win32' and value.startswith('/') and not value.startswith('//'))):
                p.error(f'{key} belongs to another operating system; edit the YAML for this host')
            values[key] = path
        elif type(value) is not type(default) and not (type(default) is float and type(value) is int):
            p.error(f'Invalid type for {key}')
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < (0 if key in ('seed', 'boundary_margin', 'erosion_margin') else 1e-12):
                p.error(f'Invalid range for {key}')
    if (values['process_res'], values['device'], values['precision'], values['offline']) != (560, 'cuda:0', 'bf16', True):
        p.error('Paired experiment requires process_res=560, cuda:0, bf16, offline=true')
    if values['expected_samples'] != 20:
        p.error('This experiment must reuse exactly the original 20 manifest samples')
    if values['boundary_margin'] >= values['ring_radius'] or not 0 < values['compression_ratio'] < 1:
        p.error('Invalid local geometry parameters')
    if values['data_root'] is None and not a.print_config:
        p.error('Set data_root in the paired YAML to the real server dataset path')
    for key, value in values.items():
        setattr(a, key, value)
    return a
