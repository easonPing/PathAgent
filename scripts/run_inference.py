"""One GPU process with atomic results, strict resume identity and complete trajectories."""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (atomic_json, code_hash, digest, load_config, read_json,
                                    read_manifest, resolve, sample_seed, seed_everything)
from data_processing.preprocess import prepare_observations, prepare_regions
from eval.metrics import score
from models.agent import run_agent
from models.inference import ModelBackend, ModelProtocolError


def check_assets(config):
    for role, model in config['models'].items():
        path = resolve(model['path'])
        lock = read_json(path / 'asset_lock.json')
        if not lock.get('complete') or lock['revision'] != model['revision'] or lock['repo'] != model['repo']:
            raise ValueError(f'{role}: model lock does not match frozen configuration')
        for name, info in lock['files'].items():
            file = path / name
            if not file.is_file() or file.stat().st_size != info['bytes']:
                raise ValueError(f'{role}: missing or truncated model file {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/reproduce.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config, manifest = load_config(args.config), read_manifest(args.manifest)
    if not manifest.get('images_verified'):
        raise ValueError('Annotations-only manifests cannot run inference')
    if args.smoke != (manifest.get('purpose') == 'smoke'):
        raise ValueError('--smoke must match the manifest purpose')
    if manifest['dataset'] == 'wsi_vqa' and not args.smoke:
        raise ValueError('This approved execution scope enables WSI-VQA smoke only')
    source_hash = code_hash()
    if os.environ.get('PATHAGENT_EXPECT_CODE_HASH', source_hash) != source_hash:
        raise ValueError('Source files changed since job submission')
    identity = {'config': config, 'code_hash': source_hash, 'samples_hash': manifest['samples_hash']}
    run_hash = digest(identity)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'run_lock.json'
    if lock_path.exists() and read_json(lock_path)['run_hash'] != run_hash:
        raise ValueError('Resume refused: code/config/manifest changed; use a new output directory')
    atomic_json(lock_path, {**identity, 'run_hash': run_hash})
    check_assets(config)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU with native BF16 is required')
    expected_gpu = os.environ.get('PATHAGENT_EXPECT_GPU_NAME')
    expected_memory = os.environ.get('PATHAGENT_EXPECT_GPU_MEMORY_BYTES')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    fleet_path = os.environ.get('PATHAGENT_GPU_FLEET_LOCK')
    if fleet_path:
        import fcntl
        fleet = Path(fleet_path)
        fleet.parent.mkdir(parents=True, exist_ok=True)
        with open(str(fleet) + '.lock', 'a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            spec = {'gpu': gpu_name, 'gpu_memory_bytes': gpu_memory}
            if fleet.exists() and read_json(fleet) != spec:
                raise RuntimeError('Benchmark shards must use identical GPU model and memory spec')
            atomic_json(fleet, spec)
    if expected_gpu and gpu_name != expected_gpu:
        raise RuntimeError(f'GPU model differs from benchmark lock: {gpu_name} != {expected_gpu}')
    if expected_memory and gpu_memory != int(expected_memory):
        raise RuntimeError('GPU memory spec differs from benchmark lock')
    started = time.monotonic()
    groups = defaultdict(list)
    for row in manifest['samples']:
        groups[row['image_id']].append(row)
    # Finish segmentation in subprocesses before resident LLMs consume GPU memory.
    prepared = {}
    for image in sorted(groups):
        prepared[image] = prepare_regions(groups[image][0], config, source_hash)
    seed_everything(config['seed'])
    backend = ModelBackend(config)
    atomic_json(output / 'resolved_generation.json', backend.resolved_generation)
    results, costs = {}, {}
    for image in sorted(groups):
        image_started = time.monotonic()
        directory, regions = prepared[image]
        descriptions, features = prepare_observations(directory, regions, backend, config)
        for row in sorted(groups[image], key=lambda r: r['sample_id']):
            destination = output / 'results' / (digest(row['sample_id']) + '.json')
            if destination.exists():
                result = read_json(destination)
                if result.get('run_hash') != run_hash or result.get('sample_id') != row['sample_id']:
                    raise ValueError('Result identity mismatch')
                if result.get('status') == 'ok':
                    results[row['sample_id']] = result
                    continue
            backend.records.clear()
            backend.set_case(row['sample_id'])
            seed_everything(sample_seed(config['seed'], row['sample_id']))
            try:
                result = run_agent(row, regions, descriptions, features, backend, config)
            except ModelProtocolError as exc:
                result = {'status': 'invalid', 'pred_answer': '', 'error': str(exc)}
            result.update(sample_id=row['sample_id'], image_id=image, run_hash=run_hash,
                          model_calls=list(backend.records))
            atomic_json(destination, result)
            results[row['sample_id']] = result
            print(f"{row['sample_id']}: {result['status']}", flush=True)
        costs[image] = {'seconds': time.monotonic() - image_started, 'regions': len(regions),
                        'questions': len(groups[image])}
    metrics = score(manifest['samples'], results, open_metrics=manifest['dataset'] == 'wsi_vqa')
    atomic_json(output / 'metrics.json', metrics)
    profile = {'real_gpu': True, 'gpu': gpu_name, 'gpu_memory_bytes': gpu_memory,
               'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
               'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
               'seconds': time.monotonic() - started, 'model_load_seconds': backend.load_seconds,
               'image_costs': costs, 'run_hash': run_hash, 'code_hash': source_hash,
               'config_hash': digest(config), 'manifest_hash': manifest['samples_hash'],
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               'smoke': args.smoke, 'dataset': manifest['dataset'],
               'availability_restricted': manifest.get('availability_restricted', False),
               'source_coverage': manifest.get('summary', {}).get('sources', {}),
               'passed': metrics['complete'] and metrics['invalid'] == 0}
    atomic_json(output / 'profile.json', profile)
    backend.close()
    if args.smoke and not profile['passed']:
        raise RuntimeError('Real GPU smoke failed: missing or invalid results')


if __name__ == '__main__':
    main()
