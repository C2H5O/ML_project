"""Inspect all paired files, without model imports or dataset writes."""
import argparse
import _bootstrap
from src.datasets.cleargrasp import inspect_dataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', default='results/cleargrasp_zero_shot')
    a = p.parse_args()
    r = inspect_dataset(a.data_root, a.output_dir)
    print(r['data_root'])
    for name, info in r['splits'].items():
        print(name, info)
    print('EXR:', r['exr_capabilities'])
    print('Missing:', len(r['missing_files']), 'Unreadable:', len(r['unreadable_files']))
    print('Saved dataset_inspection.json and sample_manifest.json in', a.output_dir)
    return 1 if r['missing_files'] or r['unreadable_files'] or r['orphan_annotations'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
