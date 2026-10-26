"""Validated ClearGrasp identities and readable/legacy result paths.

The shared src.common.sample_output remains unchanged for other datasets.
"""
import re
import stat
from pathlib import Path

from src.common import read_json, sample_output

SPLITS = {'real-test', 'real-val', 'synthetic-test', 'synthetic-val'}
RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
            *(f'LPT{i}' for i in range(1, 10))}


def sample_parts(sample):
    split, sid = sample.get('split'), sample.get('sample_id')
    if split not in SPLITS or not isinstance(sid, str):
        raise ValueError('Invalid ClearGrasp split/sample ID')
    # IDs are POSIX-style metadata on both platforms, never OS path strings.
    parts = sid.split('/')
    if parts[0] == split:
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError(f'Expected camera-or-group/numeric-ID, optionally prefixed by split: {sid!r}')
    group, number = parts
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', group, flags=re.ASCII) or
            group.upper() in RESERVED or not re.fullmatch(r'[0-9]+', number, flags=re.ASCII)):
        raise ValueError(f'Unsafe ClearGrasp sample ID: {sid!r}')
    if split.startswith('real-') and group not in {'d415', 'd435'}:
        raise ValueError(f'Unknown real camera in sample ID: {group}')
    if 'group' in sample and sample['group'] != group:
        raise ValueError('Manifest group disagrees with sample ID')
    if 'number' in sample and sample['number'] != number:
        raise ValueError('Manifest number disagrees with sample ID')
    return split, group, number


def full_sample_id(sample):
    return '/'.join(sample_parts(sample))


def safe_result_path(root, *parts):
    """Containment and link checks also cover existing directory components."""
    root = Path(root).absolute()
    path = root.joinpath(*parts)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError('Result path escapes output root') from exc
    if any(p in ('.', '..') or '\\' in p or ':' in p for p in relative.parts):
        raise ValueError('Unsafe result path component')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Result path resolves outside output root')
    current = root
    for component in ('', *relative.parts):
        if component:
            current = current / component
        if current.is_symlink():
            raise ValueError(f'Symbolic links are not supported in result paths: {current}')
        if current.exists():
            attributes = getattr(current.lstat(), 'st_file_attributes', 0)
            if attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
                raise ValueError(f'Reparse points are not supported in result paths: {current}')
    return path


def cleargrasp_prediction_dir(root, sample):
    return safe_result_path(root, 'predictions', *sample_parts(sample))


def cleargrasp_figure_dir(root, sample):
    return safe_result_path(root, 'figures', *sample_parts(sample))


def legacy_prediction_dirs(root, sample):
    split, group, number = sample_parts(sample)
    # Older manifests may store either relative IDs or split-prefixed IDs.
    ids = dict.fromkeys((sample['sample_id'], f'{group}/{number}', f'{split}/{group}/{number}'))
    return [safe_result_path(root, 'predictions', split, sample_output(root, split, sid).name) for sid in ids]


def resolve_prediction_dir(root, sample, allow_missing=False):
    candidates = [cleargrasp_prediction_dir(root, sample), *legacy_prediction_dirs(root, sample)]
    present = [p for p in candidates if p.exists()]
    if len(present) > 1:
        raise ValueError(f'Ambiguous readable/legacy predictions for {full_sample_id(sample)}: {present}')
    if present:
        if not present[0].is_dir():
            raise ValueError(f'Prediction path is not a directory: {present[0]}')
        return present[0]
    if allow_missing:
        return candidates[0]
    raise FileNotFoundError(f'No saved prediction for {full_sample_id(sample)}; checked {candidates}')


def legacy_figure_dirs(root, sample):
    return [safe_result_path(root, 'figures', sample['split'], p.name)
            for p in legacy_prediction_dirs(root, sample)]


def load_result_manifest(root, expected_samples=None):
    path = safe_result_path(root, 'sample_manifest.json')
    manifest = read_json(path)
    if manifest.get('dataset') != 'ClearGrasp':
        raise ValueError('This operation requires a ClearGrasp manifest')
    samples = manifest.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('Manifest has no saved sample selection')
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(f'Expected all {expected_samples} saved samples; manifest contains {len(samples)}')
    identities = [full_sample_id(s).casefold() for s in samples]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate sample identities in manifest (including case-insensitive collisions)')
    return manifest
