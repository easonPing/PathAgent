"""Submit smoke or both final 16-shard arrays, with prerequisite attestations."""
import argparse
import datetime as dt
import math
import shlex
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, code_hash, digest, load_config,
                                    read_json, read_manifest, resolve, write_manifest)
from data_processing.regions import ImageSource
from data_processing.sharding import make_shards
from scripts.run_inference import check_assets
from scripts.select_gpu import GPU_PRIORS, choose_pair, discover


def submit(script, candidate, config, name, array=False):
    command = ['sbatch', '--parsable', '--account=' + config['slurm']['account'], '--partition=gpu',
               '--gres=gpu:' + candidate['gres'] + ':1', '--constraint=' + candidate['feature'],
               '--cpus-per-task=' + str(config['slurm']['cpus_per_task']),
               '--mem=' + str(config['slurm']['memory_gb']) + 'G',
               '--time=' + str(candidate['walltime_minutes']), '--job-name=' + name,
               '--output=' + str(script.parent / (name + '-%A_%a.log'))]
    if array:
        command.append('--array=0-15%16')
    result = subprocess.run(command + [str(script)], check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError('Slurm submission response lacks a numeric job ID: ' + result.stdout)
    # Persist immediately, including when later verification/submission fails.
    atomic_json(script.with_suffix('.submitted.json'), {'job_id': job_id, 'command': command})
    verify = subprocess.run(['scontrol', 'show', 'job', job_id, '-o'], check=True, capture_output=True, text=True)
    if 'JobId=' not in verify.stdout:
        raise RuntimeError(f'Cannot verify submitted job {job_id}')
    return {'job_id': job_id, 'command': command, 'verification': verify.stdout, 'candidate': candidate}


def script_text(manifest, output, config_path, source_hash, smoke=False, fleet=None, array=False):
    py = shlex.quote(sys.executable)
    lines = ['#!/bin/bash', 'set -euo pipefail', 'cd ' + shlex.quote(str(ROOT)),
             'export TOKENIZERS_PARALLELISM=false', 'export CUBLAS_WORKSPACE_CONFIG=:4096:8',
             'export PATHAGENT_EXPECT_CODE_HASH=' + shlex.quote(source_hash)]
    if fleet:
        lines.append('export PATHAGENT_GPU_FLEET_LOCK=' + shlex.quote(str(fleet)))
    if array:
        lines.extend(['shard_index="${SLURM_ARRAY_TASK_ID:?Missing array index}"',
                      'manifest=' + shlex.quote(str(manifest)) + '/shard_${shard_index}.json',
                      'output=' + shlex.quote(str(output)) + '/shard_${shard_index}'])
    else:
        lines.extend(['manifest=' + shlex.quote(str(manifest)), 'output=' + shlex.quote(str(output))])
    lines.append('exec ' + py + ' pathagent.py --config ' + shlex.quote(str(config_path)) +
                 ' --manifest "$manifest" --output "$output"' + (' --smoke' if smoke else ''))
    return '\n'.join(lines) + '\n'


def verify_smoke(path, dataset, config, source_hash):
    path = resolve(path)
    profile = read_json(path / 'profile.json')
    manifest = read_manifest(f'data/manifests/{dataset}/smoke.json')
    if manifest.get('availability_restricted'):
        raise RuntimeError('Availability-restricted checks do not replace complete-source smoke acceptance')
    lock = read_json(path / 'run_lock.json')
    expected = digest({'config': config, 'code_hash': source_hash, 'samples_hash': manifest['samples_hash']})
    if not (profile.get('passed') and profile.get('real_gpu') and profile.get('smoke')
            and profile.get('dataset') == dataset and profile.get('slurm_job_id')
            and profile['code_hash'] == source_hash and profile['config_hash'] == digest(config)
            and profile['manifest_hash'] == manifest['samples_hash']
            and profile['run_hash'] == expected and lock['run_hash'] == expected):
        raise RuntimeError(f'{dataset}: missing or stale successful real-GPU smoke attestation')
    results = [read_json(path / 'results' / (digest(row['sample_id']) + '.json')) for row in manifest['samples']]
    if any(r.get('status') != 'ok' or r.get('run_hash') != expected for r in results):
        raise RuntimeError(f'{dataset}: smoke results failed integrity checks')
    return profile


def image_costs(samples, profile):
    units = sum(r['regions'] + r['questions'] * 5 for r in profile['image_costs'].values())
    seconds_per_unit = max(1, profile['seconds'] - profile['model_load_seconds']) / units
    model = profile['gpu'].lower()
    feature = next((k for k in GPU_PRIORS if k in model.replace(' ', '_')), None)
    if 'a100' in model:
        feature = 'a100_80gb' if profile['gpu_memory_bytes'] > 50 * 1024 ** 3 else 'a100_40gb'
    if 'rtx' in model and '6000' in model and 'pro' in model:
        feature = 'rtxpro6000'
    if feature is None:
        raise ValueError('Cannot normalize smoke GPU runtime to hardware prior')
    seconds_per_unit *= GPU_PRIORS[feature]['speed']
    groups = {}
    for row in samples:
        groups.setdefault(row['image_id'], []).append(row)
    costs = {}
    for image, rows in groups.items():
        with ImageSource(rows[0]['image_path'], rows[0]['assumed_mpp']) as src:
            n = 1 if rows[0]['input_kind'] == 'roi' else math.ceil(src.width / 4096) * math.ceil(src.height / 4096)
        costs[image] = seconds_per_unit * (n + len(rows) * 5)
    return costs


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'final'])
    parser.add_argument('--dataset', choices=['pathmmu', 'bcnb', 'wsi_vqa'])
    parser.add_argument('--manifest', help='Explicit smoke manifest, including availability-restricted execution checks')
    parser.add_argument('--smoke-pathmmu')
    parser.add_argument('--smoke-bcnb')
    parser.add_argument('--smoke-wsi-vqa')
    parser.add_argument('--estimate-seconds', type=float, default=14400,
                        help='Initial smoke runtime prior in A100-80GB seconds; not a measurement')
    args = parser.parse_args()
    config = load_config()
    check_assets(config)
    source_hash = code_hash()
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root = resolve('runs/submissions') / stamp
    root.mkdir(parents=True)
    config_path = root / 'reproduce.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.mode == 'smoke':
        if not args.dataset:
            parser.error('--dataset is required for smoke')
        manifest_path = resolve(args.manifest or f'data/manifests/{args.dataset}/smoke.json')
        manifest = read_manifest(manifest_path)
        if manifest['dataset'] != args.dataset:
            raise ValueError('Smoke manifest dataset differs from --dataset')
        if manifest.get('purpose') != 'smoke' or not manifest.get('images_verified'):
            raise ValueError('Need a validated real-image smoke manifest')
        report = discover(config, [args.estimate_seconds], minimum_memory_gb=40)
        atomic_json(root / 'gpu_selection.json', report)
        if not report['candidates']:
            raise RuntimeError('No eligible GPU candidate')
        script = root / 'smoke.sh'
        output = root / args.dataset
        script.write_text(script_text(manifest_path, output, config_path, source_hash, smoke=True))
        receipt = submit(script, report['candidates'][0], config, 'pathagent-smoke-' + args.dataset)
        atomic_json(root / 'receipt.json', {'output': str(output), **receipt})
        print({'job_id': receipt['job_id'], 'output': str(output)})
        return
    profiles = {}
    for dataset in ['pathmmu', 'bcnb', 'wsi_vqa']:
        path = getattr(args, 'smoke_' + dataset)
        if not path:
            parser.error('Final arrays require all three --smoke-* output paths')
        profiles[dataset] = verify_smoke(path, dataset, config, source_hash)
    manifests, costs, reports = {}, {}, {}
    for dataset in ['pathmmu', 'bcnb']:
        full = read_manifest(f'data/manifests/{dataset}/full.json')
        if full.get('purpose') != 'full' or not full.get('images_verified'):
            raise ValueError('Full benchmark needs a complete image-verified manifest')
        from data_processing.datasets import load_benchmark
        fresh = load_benchmark(full['dataset_config'], require_images=True)
        if digest(fresh) != full['samples_hash']:
            raise ValueError('Full dataset differs from manifest')
        estimated = image_costs(full['samples'], profiles[dataset])
        shards, totals = make_shards(full['samples'], 16, estimated)
        manifests[dataset], costs[dataset] = shards, totals
        for i, rows in enumerate(shards):
            write_manifest(root / dataset / 'manifests' / f'shard_{i}.json', rows,
                           dataset=dataset, purpose='shard', images_verified=True,
                           parent_samples_hash=full['samples_hash'], shard_index=i, shard_count=16)
        atomic_json(root / dataset / 'costs.json', {'method': 'smoke seconds per region+5*QA; full-image grid upper bound',
                                                  'image_costs': estimated, 'shard_costs': totals})
        peak_gb = profiles[dataset]['max_memory_reserved_bytes'] / 1024 ** 3
        reports[dataset] = discover(config, totals, max(32, peak_gb * config['slurm']['gpu_memory_headroom']))
        atomic_json(root / dataset / 'gpu_selection.json', reports[dataset])
    pair = choose_pair(reports['pathmmu'], reports['bcnb'], costs['pathmmu'], costs['bcnb'])
    atomic_json(root / 'pair_selection.json', pair)
    receipts = {}
    for dataset in ['pathmmu', 'bcnb']:
        directory = root / dataset
        script = directory / 'array.sh'
        script.write_text(script_text(directory / 'manifests', directory / 'outputs', config_path,
                                      source_hash, fleet=directory / 'gpu_fleet.json', array=True))
        receipts[dataset] = submit(script, pair[dataset], config, 'pathagent-' + dataset, array=True)
        atomic_json(root / 'receipts.json', receipts)
        print({'dataset': dataset, 'job_id': receipts[dataset]['job_id'], 'shards': 16}, flush=True)
    print('Both arrays accepted. Receipt:', root / 'receipts.json')


if __name__ == '__main__':
    main()
