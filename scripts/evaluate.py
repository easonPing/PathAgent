"""Merge atomic shard results against one full manifest without changing its denominator."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, read_json, read_manifest, resolve
from eval.metrics import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    results, config_hashes, source_hashes = {}, set(), set()
    for path in sorted(resolve(args.results_dir).rglob('results/*.json')):
        row = read_json(path)
        identity = row['sample_id']
        if identity in results:
            raise ValueError(f'Duplicate result: {identity}')
        lock = read_json(path.parent.parent / 'run_lock.json')
        from data_processing.common import digest
        if row['run_hash'] != lock['run_hash']:
            raise ValueError('Result run hash differs from its run lock')
        config_hashes.add(digest(lock['config']))
        source_hashes.add(lock['code_hash'])
        results[identity] = row
    if len(config_hashes) > 1 or len(source_hashes) > 1:
        raise ValueError('Cannot merge runs with different configurations or source code')
    metrics = score(manifest['samples'], results, open_metrics=manifest['dataset'] == 'wsi_vqa')
    atomic_json(resolve(args.output), metrics)
    print(metrics)
    if not metrics['complete'] and not args.allow_incomplete:
        raise SystemExit('Incomplete run: metrics saved with full manifest denominator; not a final reproduction result')


if __name__ == '__main__':
    main()
