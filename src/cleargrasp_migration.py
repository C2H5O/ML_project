"""Manifest-driven, no-clobber directory migration; prediction bytes are immutable."""
from contextlib import contextmanager
from pathlib import Path

from src.common import read_json, write_json, digest_file
from src.datasets.cleargrasp_paths import (load_result_manifest, full_sample_id,
    cleargrasp_prediction_dir, cleargrasp_figure_dir, legacy_prediction_dirs,
    legacy_figure_dirs, resolve_prediction_dir, safe_result_path)
from src.evaluation.cleargrasp import saved_prediction

JOURNAL = 'path_migration.json'
LOCK = '.cleargrasp_migration.lock'


def tree_hashes(root, folder):
    folder = safe_result_path(root, Path(folder).relative_to(Path(root).absolute()))
    hashes = {}
    for p in sorted(folder.rglob('*')):
        checked = safe_result_path(root, p.relative_to(Path(root).absolute()))
        if checked.is_file():
            hashes[checked.relative_to(folder).as_posix()] = digest_file(checked)
    return hashes


def plan_migration(output, expected_samples=20):
    root = Path(output).absolute()
    manifest = load_result_manifest(root, expected_samples)
    operations, mapping = [], []
    # Validate EVERY prediction before planning any write or moving any directory.
    for sample in manifest['samples']:
        saved_prediction(root, sample, require_metric=True)
        source = resolve_prediction_dir(root, sample)
        destination = cleargrasp_prediction_dir(root, sample)
        item = dict(sample_id=full_sample_id(sample),
                    prediction_before=source.relative_to(root).as_posix(),
                    prediction_after=destination.relative_to(root).as_posix())
        candidates = [cleargrasp_figure_dir(root, sample), *legacy_figure_dirs(root, sample)]
        existing = [p for p in candidates if p.exists()]
        if len(existing) > 1 or any(not p.is_dir() for p in existing):
            raise ValueError(f'Ambiguous/conflicting figures for {full_sample_id(sample)}')
        figures_source = existing[0] if existing else None
        figures_dest = candidates[0]
        item.update(figures_before=figures_source.relative_to(root).as_posix() if figures_source else None,
                    figures_after=figures_dest.relative_to(root).as_posix(),
                    figures_status='present' if figures_source else 'absent; no files invented')
        mapping.append(item)
        for kind, src, dst in [('predictions', source, destination), ('figures', figures_source, figures_dest)]:
            if src is None or src == dst:
                continue
            if dst.exists():
                raise FileExistsError(f'Refusing to overwrite migration destination: {dst}')
            operations.append(dict(sample_id=full_sample_id(sample), kind=kind,
                source=src.relative_to(root).as_posix(), destination=dst.relative_to(root).as_posix(),
                sha256=tree_hashes(root, src)))
    return dict(schema_version=1, dataset='ClearGrasp', samples=len(manifest['samples']),
                manifest_sha256=digest_file(root/'sample_manifest.json'), mapping=mapping,
                operations=operations, state='planned', completed=[])


@contextmanager
def migration_lock(root):
    lock = safe_result_path(root, LOCK)
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError(f'Migration lock exists: {lock}. Ensure no migration/inference is running. '
                           'If the previous process crashed, remove only this lock file and run --recover.') from exc
    try:
        with stream:
            stream.write('Exclusive ClearGrasp migration. Do not run inference, plotting or analysis concurrently.\n')
        yield
    finally:
        lock.unlink()


def _rename_checked(root, src, dst):
    src = safe_result_path(root, src)
    dst = safe_result_path(root, dst)
    if dst.exists():
        raise FileExistsError(f'Refusing to overwrite: {dst}')
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Paths have been resolved/contained and checked for links before directory move.
    safe_result_path(root, dst.relative_to(Path(root).absolute()))
    src.rename(dst)


def _validate_journal(root, journal):
    manifest = load_result_manifest(root, journal.get('samples'))
    if digest_file(Path(root)/'sample_manifest.json') != journal.get('manifest_sha256'):
        raise ValueError('Manifest changed since migration was planned; recovery requires manual review')
    allowed = {}
    for sample in manifest['samples']:
        sid = full_sample_id(sample)
        allowed[(sid, 'predictions')] = (legacy_prediction_dirs(root, sample), cleargrasp_prediction_dir(root, sample))
        allowed[(sid, 'figures')] = (legacy_figure_dirs(root, sample), cleargrasp_figure_dir(root, sample))
    seen = set()
    for op in journal['operations']:
        key = (op['sample_id'], op['kind'])
        if key not in allowed or key in seen:
            raise ValueError('Invalid/duplicate journal operation')
        seen.add(key)
        sources, target = allowed[key]
        if safe_result_path(root, op['source']) not in sources or safe_result_path(root, op['destination']) != target:
            raise ValueError('Journal paths do not match the manifest-derived migration')
        if not isinstance(op.get('sha256'), dict):
            raise ValueError('Missing migration file checksums')


def _rollback(root, journal):
    errors = []
    for op in reversed(journal['operations']):
        src, dst = safe_result_path(root, op['source']), safe_result_path(root, op['destination'])
        try:
            if src.exists() and not dst.exists():
                if tree_hashes(root, src) != op['sha256']:
                    raise ValueError('Original directory content changed; refusing recovery')
                continue
            if not src.exists() and dst.exists():
                if tree_hashes(root, dst) != op['sha256']:
                    raise ValueError('Moved directory content changed; refusing recovery')
                _rename_checked(root, op['destination'], op['source'])
            else:
                raise ValueError('Both source/destination exist, or both are absent; refusing to guess')
        except Exception as exc:
            errors.append(dict(sample_id=op['sample_id'], kind=op['kind'], error=str(exc)))
    journal.update(state='rollback_incomplete' if errors else 'rolled_back', rollback_errors=errors,
        recovery='Stop concurrent writers. Preserve both trees and this journal. After resolving conflicts, '
                 'run migrate_cleargrasp_result_paths.py --config <same YAML> --recover. '
                 'Recovery never overwrites files and verifies planned checksums.')
    write_json(Path(root)/JOURNAL, journal)
    return errors


def migrate_results(output, expected_samples=20, apply=False, recover=False):
    root = Path(output).absolute()
    journal_path = safe_result_path(root, JOURNAL)
    if not apply and not recover:
        plan = plan_migration(root, expected_samples)
        if journal_path.exists():
            plan['previous_state'] = read_json(journal_path).get('state')
        return plan  # Dry-run writes nothing, including no journal or lock.
    with migration_lock(root):
        if recover:
            journal = read_json(journal_path)
            if journal.get('state') not in ('applying', 'rollback_incomplete'):
                raise ValueError('Recovery requires an interrupted/incomplete migration journal')
            _validate_journal(root, journal)
            errors = _rollback(root, journal)
            if errors:
                raise RuntimeError(f'Rollback incomplete; inspect {journal_path} before retrying --recover')
            return journal
        if journal_path.exists() and read_json(journal_path).get('state') in ('applying', 'rollback_incomplete'):
            raise RuntimeError(f'Unfinished migration; inspect {journal_path} and use --recover first')
        plan = plan_migration(root, expected_samples)
        if not plan['operations']:
            plan['state'] = 'already_readable'
            return plan
        plan['state'] = 'applying'
        write_json(journal_path, plan)
        try:
            for index, op in enumerate(plan['operations']):
                if digest_file(root/'sample_manifest.json') != plan['manifest_sha256']:
                    raise ValueError('Manifest changed during migration')
                if tree_hashes(root, root/op['source']) != op['sha256']:
                    raise ValueError(f'Source changed after preflight: {op["source"]}')
                _rename_checked(root, op['source'], op['destination'])
                if tree_hashes(root, root/op['destination']) != op['sha256']:
                    raise ValueError(f'Post-move checksum mismatch: {op["destination"]}')
                plan['completed'].append(index)
                write_json(journal_path, plan)
            plan['state'] = 'completed'
            write_json(journal_path, plan)
            return plan
        except BaseException as exc:
            plan['failure'] = f'{type(exc).__name__}: {exc}'
            try:
                errors = _rollback(root, plan)
                outcome = 'rollback incomplete' if errors else 'all completed moves rolled back'
            except Exception as rollback_exc:
                outcome = f'rollback could not finish: {rollback_exc}'
            raise RuntimeError(f'Migration failed; {outcome}. Inspect {journal_path}; '
                               'use --recover for an interrupted migration. Prediction arrays were not rewritten.') from exc
