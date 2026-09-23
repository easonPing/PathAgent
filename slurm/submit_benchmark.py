"""Submit a complete benchmark when the user explicitly waives smoke acceptance.

Kept outside the inference source tree so existing queued runs retain their
source identity. The launcher itself is hashed in the submission record.
"""
import argparse
import datetime as dt
import math
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, code_hash, digest, load_config,
                                    read_manifest, resolve, sha256_file, write_manifest)
from data_processing.datasets import load_benchmark, summarize
from data_processing.regions import ImageSource
from data_processing.sharding import make_shards
from scripts.run_inference import check_assets
from scripts.select_gpu import GPU_PRIORS, discover
from scripts.submit_runs import script_text, submit


def prepare(dataset, root, region_seconds, question_seconds, shard_count=16):
    full = read_manifest(f'data/manifests/{dataset}/full.json')
    if (full.get('purpose') != 'full' or not full.get('images_verified')
            or full.get('availability_restricted')):
        raise ValueError('Only complete, image-verified full manifests are accepted')
    fresh = load_benchmark(full['dataset_config'], require_images=True)
    if digest(fresh) != full['samples_hash']:
        raise ValueError('Dataset differs from its full manifest')
    counts = Counter(row['image_id'] for row in fresh)
    costs = {}
    for row in fresh:
        image = row['image_id']
        if image in costs:
            continue
        with ImageSource(row['image_path'], row['assumed_mpp']) as source:
            if row['input_kind'] == 'wsi' and source.mpp is None:
                raise ValueError(f'Missing physical scale: {image}')
            regions = (1 if row['input_kind'] == 'roi' else
                       math.ceil(source.width / 4096) * math.ceil(source.height / 4096))
        costs[image] = regions * region_seconds + counts[image] * question_seconds
    shards, totals = make_shards(fresh, shard_count, costs)
    for index, rows in enumerate(shards):
        write_manifest(root / 'manifests' / f'shard_{index}.json', rows,
                       dataset=dataset, purpose='shard', images_verified=True,
                       availability_restricted=False, summary=summarize(rows),
                       parent_samples_hash=full['samples_hash'], shard_index=index, shard_count=shard_count)
    write_manifest(root / 'full.json', fresh, **{
        k: v for k, v in full.items() if k not in {'samples', 'samples_hash'}})
    atomic_json(root / 'costs.json', {
        'method': 'UNMEASURED A100-80GB engineering prior; grid region upper bound',
        'region_seconds': region_seconds, 'question_seconds': question_seconds,
        'image_costs': costs, 'shard_costs': totals,
        'warning': 'Smoke was waived. Actual memory and throughput have not been measured.'})
    return full, totals


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['pathmmu', 'bcnb'])
    parser.add_argument('--skip-smoke', action='store_true', required=True,
                        help='Explicit user-authorized waiver, not a successful smoke attestation')
    parser.add_argument('--submit', action='store_true', help='Actually submit; otherwise prepare only')
    parser.add_argument('--region-seconds', type=float, default=20)
    parser.add_argument('--question-seconds', type=float, default=120)
    parser.add_argument('--minimum-vram-gb', type=float, default=40)
    parser.add_argument('--shards', type=int, choices=[16, 128], default=16)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--gpu-feature', choices=sorted(GPU_PRIORS),
                        help='Require every shard to use this GPU model and memory specification')
    args = parser.parse_args()
    if min(args.region_seconds, args.question_seconds, args.minimum_vram_gb) <= 0:
        parser.error('Estimates and minimum VRAM must be positive')
    if not 1 <= args.concurrency <= min(args.shards, 32):
        parser.error('Concurrency must be between 1 and min(shards, 32)')
    config = load_config()
    config['slurm']['shards'] = args.shards
    check_assets(config)
    source_hash = code_hash()
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root = resolve('runs/submissions') / (stamp + '_' + args.dataset + '_full')
    root.mkdir(parents=True)
    full, totals = prepare(args.dataset, root, args.region_seconds, args.question_seconds, args.shards)
    config_path = root / 'reproduce.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    load_config(config_path)
    atomic_json(root / 'authorization.json', {
        'smoke_required': False, 'smoke_passed': False,
        'reason': 'User explicitly requested full benchmarks assuming smoke succeeds',
        'full_data_required': True, 'dataset': args.dataset,
        'source_hash': source_hash, 'launcher_sha256': sha256_file(__file__),
        'full_samples_hash': full['samples_hash'], 'summary': summarize(full['samples']),
        'shard_count': args.shards, 'gpu_feature': args.gpu_feature,
        'concurrency_limit': args.concurrency})
    script = root / 'array.sh'
    script.write_text(script_text(root / 'manifests', root / 'outputs', config_path,
                                  source_hash, fleet=root / 'gpu_fleet.json', array=True))
    aggregate = root / 'evaluate.sh'
    aggregate.write_text('\n'.join([
        '#!/bin/bash', 'set -euo pipefail', 'cd ' + shlex.quote(str(ROOT)),
        'exec ' + shlex.quote(sys.executable) + ' scripts/evaluate.py --manifest ' +
        shlex.quote(str(root / 'full.json')) + ' --results-dir ' + shlex.quote(str(root / 'outputs')) +
        ' --output ' + shlex.quote(str(root / 'metrics.json')), '']))
    print({'prepared': str(root), 'summary': summarize(full['samples']),
           'estimated_max_shard_a100_hours': max(totals) / 3600}, flush=True)
    if not args.submit:
        return
    report = discover(config, totals, minimum_memory_gb=args.minimum_vram_gb)
    atomic_json(root / 'gpu_selection.json', report)
    candidates = [c for c in report['candidates']
                  if args.gpu_feature is None or c['feature'] == args.gpu_feature]
    if not candidates:
        raise RuntimeError('No scheduler-supported GPU candidate; see gpu_selection.json')
    if code_hash() != source_hash:
        raise RuntimeError('Inference sources changed during preparation')
    receipt = submit(script, candidates[0], config, 'pathagent-' + args.dataset, array=True,
                     shard_count=args.shards, concurrency=args.concurrency)
    atomic_json(root / 'receipt.json', receipt)
    print({'dataset': args.dataset, 'job_id': receipt['job_id'],
           'gpu': receipt['candidate']['feature'], 'shards': args.shards,
           'concurrency': args.concurrency,
           'walltime_minutes': receipt['candidate']['walltime_minutes'],
           'receipt': str(root / 'receipt.json')}, flush=True)
    # Small CPU-only scoring job runs only after all shards succeed.
    command = ['sbatch', '--parsable', '--account=' + config['slurm']['account'],
               '--partition=standard', '--cpus-per-task=1', '--mem=4G', '--time=15',
               '--dependency=afterok:' + receipt['job_id'], '--kill-on-invalid-dep=yes',
               '--job-name=pathagent-score-' + args.dataset,
               '--output=' + str(root / 'evaluate-%j.log'), str(aggregate)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError('Unexpected scoring submission response: ' + result.stdout)
    atomic_json(root / 'evaluation_receipt.json', {'job_id': job_id, 'command': command})
    print({'evaluation_job_id': job_id}, flush=True)


if __name__ == '__main__':
    main()
