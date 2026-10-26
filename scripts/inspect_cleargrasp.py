"""Inspect all paired files, without model imports or dataset writes."""
import _bootstrap
from src.datasets.cleargrasp import inspect_dataset
from src.cleargrasp_config import ClearGraspArgumentParser, print_config

def main():
    p = ClearGraspArgumentParser(description=__doc__)
    a = p.parse_args()
    if a.print_config:
        print_config(a)
        return 0
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
