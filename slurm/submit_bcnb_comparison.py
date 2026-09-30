"""Freeze and submit two BCNB validations; submit full arrays only if both pass."""
import argparse
import datetime as dt
import fcntl
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, code_hash, digest, load_config,
                                    read_json, read_manifest, resolve, run_identity,
                                    sha256_file, write_manifest)

ACCOUNT = 'cs6501-cbx8wm'
PIPELINES = ('legacy', 'paper')


def command(root, name, *, gpu=False, array=False, minutes=10, dependency=None):
    cmd = ['sbatch', '--parsable', '--account=' + ACCOUNT, '--qos=class',
           '--partition=' + ('gpu' if gpu else 'standard'),
           '--cpus-per-task=' + ('8' if gpu else '1'),
           '--mem=' + ('64G' if gpu else '4G'), '--time=' + str(minutes),
           '--job-name=bcnb-' + name,
           '--output=' + str(root / 'logs' / (name + '-%A_%a.log'))]
    if gpu:
        cmd += ['--gres=gpu:rtx_pro_6000:1', '--constraint=rtxpro6000']
    if array:
        cmd += ['--array=0-15']  # Intentionally no user-imposed concurrency cap.
    if dependency:
        cmd += ['--dependency=' + dependency, '--kill-on-invalid-dep=yes']
    return cmd + [str(root / (name + '.sh'))]


def submit_once(root, name, cmd):
    receipt = root / 'receipts' / (name + '.json')
    if receipt.exists():
        previous = read_json(receipt)
        if previous['command'] != cmd:
            raise RuntimeError('Submission parameters changed: ' + name)
        return previous['job_id']
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected sbatch response: ' + result.stdout)
    atomic_json(receipt, {'job_id': job, 'command': cmd,
                         'submitted_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    print(name, job, flush=True)
    return job


def shell(source, args, env=None):
    lines = ['#!/bin/bash', 'set -euo pipefail', 'cd ' + shlex.quote(str(source))]
    for key, value in (env or {}).items():
        lines.append('export ' + key + '=' + shlex.quote(str(value)))
    return '\n'.join(lines) + '\nexec ' + shlex.join([sys.executable] + args) + '\n'


def prepare(root):
    import yaml
    from data_processing.datasets import load_benchmark
    from data_processing.sharding import make_shards
    from scripts.build_manifest import select_smoke
    from scripts.run_inference import check_assets
    from scripts.submit_runs import script_text

    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    full = read_manifest('data/manifests/bcnb/full.json')
    if (full['dataset'] != 'bcnb' or full['purpose'] != 'full'
            or not full.get('images_verified') or full.get('availability_restricted')):
        raise ValueError('Complete image-verified BCNB manifest required')
    fresh = load_benchmark(full['dataset_config'], require_images=True)
    if digest(fresh) != full['samples_hash']:
        raise ValueError('BCNB dataset changed since manifest creation')
    atomic_json(root / 'full.json', full)
    smoke = select_smoke(fresh, 'bcnb')
    write_manifest(root / 'smoke.json', smoke, dataset='bcnb', purpose='smoke',
                   images_verified=True, availability_restricted=False, full_source_smoke=True,
                   parent_samples_hash=full['samples_hash'])
    history = read_json(resolve('runs/planning/bcnb_cs6501_20260928/historical_costs.json'))
    # Same image partition for both pipelines, balanced using measured historical work.
    costs = {image: value['seconds'] + history['region_costs'][image]['seconds']
             for image, value in history['image_costs'].items()}
    if set(costs) != {row['image_id'] for row in fresh}:
        raise ValueError('Historical cost coverage differs from current BCNB dataset')
    shards, totals = make_shards(fresh, 16, costs)
    for index, rows in enumerate(shards):
        write_manifest(root / 'manifests' / f'shard_{index}.json', rows,
                       dataset='bcnb', purpose='shard', images_verified=True,
                       availability_restricted=False, parent_samples_hash=full['samples_hash'],
                       shard_index=index, shard_count=16)
    atomic_json(root / 'sharding.json', {'method': 'LPT historical per-image inference + segmentation seconds',
                                       'cost_source': '20260923_105000_254981_bcnb_full',
                                       'shard_costs': totals, 'same_partition_both_pipelines': True})
    source = root / 'source'
    files = [ROOT / 'pathagent.py', Path(__file__).resolve()]
    for directory in ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval'):
        files += list((ROOT / directory).rglob('*.py'))
    for file in files:
        dest = source / file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    source_hash = code_hash()
    plan = {'root': str(root), 'source': str(source), 'source_hash': source_hash,
            'account': ACCOUNT, 'qos': 'class', 'gpu_feature': 'rtxpro6000',
            'shards_per_pipeline': 16, 'array_throttle': None,
            'gate_rule': 'BOTH validations must pass before ANY full array is submitted',
            'validation_questions': len(smoke), 'full_questions': len(fresh),
            'full_images': len(costs), 'pipelines': {}, 'files': {}}
    for name in PIPELINES:
        config = load_config('configs/reproduce_v2.yaml' if name == 'paper' else 'configs/reproduce_v1.yaml')
        config['slurm']['account'] = ACCOUNT
        for model in config['models'].values():
            model['path'] = str(resolve(model['path']))
        config['preprocessing']['segmentation_model']['path'] = str(resolve(config['preprocessing']['segmentation_model']['path']))
        for key in ('trident_path', 'trident_python'):
            config['preprocessing'][key] = str(resolve(config['preprocessing'][key]))
        for key in ('processed_root', 'manifest_root'):
            config['runtime'][key] = str(resolve(config['runtime'][key]))
        check_assets(config)
        config_path = root / (name + '.yaml')
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_config(config_path)
        plan['pipelines'][name] = {'config': str(config_path), 'config_hash': digest(config),
                                   'full_minutes': 210 if name == 'legacy' else 294}
        for stage in ('validation', 'full'):
            text = script_text(root / ('smoke.json' if stage == 'validation' else 'manifests'),
                               root / name / stage, config_path, source_hash,
                               smoke=stage == 'validation', array=stage == 'full',
                               fleet=root / 'gpu_fleet.json')
            text = text.replace('cd ' + shlex.quote(str(ROOT)), 'cd ' + shlex.quote(str(source)), 1)
            text = text.replace('export TOKENIZERS_PARALLELISM=false',
                                'export TORCH_HOME=' + shlex.quote(str(ROOT / 'checkpoints/torch')) +
                                '\nexport TOKENIZERS_PARALLELISM=false', 1)
            (root / (name + '-' + stage + '.sh')).write_text(text)
        (root / (name + '-score.sh')).write_text(shell(source, ['scripts/evaluate.py',
            '--manifest', str(root / 'full.json'), '--results-dir', str(root / name / 'full'),
            '--output', str(root / name / 'metrics.json')]))
    (root / 'gate.sh').write_text(shell(source, ['slurm/submit_bcnb_comparison.py', 'gate', '--root', str(root)]))
    for file in [root / 'full.json', root / 'smoke.json', *sorted((root / 'manifests').glob('*.json')),
                 *root.glob('*.yaml'), *root.glob('*.sh'), source / 'slurm/submit_bcnb_comparison.py']:
        plan['files'][str(file)] = sha256_file(file)
    atomic_json(root / 'plan.json', plan)
    print(root, flush=True)


def verify_validation(plan, name, job_id):
    root = Path(plan['root'])
    config = load_config(plan['pipelines'][name]['config'])
    if digest(config) != plan['pipelines'][name]['config_hash']:
        raise RuntimeError(name + ': config changed')
    manifest = read_manifest(root / 'smoke.json')
    output = root / name / 'validation'
    profile = read_json(output / 'profile.json')
    identity = run_identity(config, plan['source_hash'], manifest['samples_hash'])
    run_hash = digest(identity)
    lock = read_json(output / 'run_lock.json')
    required = {'passed': True, 'real_gpu': True, 'smoke': True, 'dataset': 'bcnb',
                'code_hash': plan['source_hash'], 'config_hash': digest(config),
                'manifest_hash': manifest['samples_hash'], 'run_hash': run_hash}
    if any(profile.get(k) != v for k, v in required.items()) or str(profile.get('slurm_job_id')) != str(job_id):
        raise RuntimeError(name + ': missing or stale successful real-GPU validation')
    if lock != {**identity, 'run_hash': run_hash}:
        raise RuntimeError(name + ': validation run identity mismatch')
    expected_gpu = read_json(root / 'gpu_fleet.json')
    if any(profile.get(k) != v for k, v in expected_gpu.items()):
        raise RuntimeError(name + ': GPU differs from shared fleet lock')
    if 'rtx' not in profile['gpu'].lower() or 'pro' not in profile['gpu'].lower() or '6000' not in profile['gpu']:
        raise RuntimeError('Validation did not use RTX PRO 6000')
    for row in manifest['samples']:
        result = read_json(output / 'results' / (digest(row['sample_id']) + '.json'))
        if (result.get('sample_id') != row['sample_id'] or result.get('status') != 'ok'
                or result.get('run_hash') != run_hash):
            raise RuntimeError(name + ': invalid or missing validation result')
    return profile


def completed_successfully(job_ids):
    # Accounting may briefly lag the dependency release. Never infer success from absence.
    for attempt in range(7):
        result = subprocess.run(['sacct', '-X', '-j', ','.join(job_ids), '-n', '-P',
                                 '--format=JobIDRaw,State,ExitCode'], check=True, capture_output=True, text=True)
        states = {parts[0]: parts[1:3] for line in result.stdout.splitlines()
                  if len(parts := line.split('|')) >= 3}
        if all(states.get(job) == ['COMPLETED', '0:0'] for job in job_ids):
            return
        if attempt < 6:
            time.sleep(10)
    raise RuntimeError('Validation Slurm jobs did not both complete successfully: ' + repr(states))


def gate(root):
    plan = read_json(root / 'plan.json')
    try:
        if code_hash() != plan['source_hash']:
            raise RuntimeError('Frozen inference sources changed')
        for path, expected in plan['files'].items():
            if sha256_file(path) != expected:
                raise RuntimeError('Frozen input changed: ' + path)
        jobs = {name: read_json(root / 'receipts' / (name + '-validation.json'))['job_id']
                for name in PIPELINES}
        completed_successfully(list(jobs.values()))
        profiles = {name: verify_validation(plan, name, jobs[name]) for name in PIPELINES}
    except Exception as exc:
        atomic_json(root / 'gate_decision.json', {'passed': False, 'full_submitted': False, 'reason': str(exc)})
        raise
    atomic_json(root / 'gate_decision.json', {'passed': True, 'validation_jobs': jobs,
                'profiles': profiles, 'decision': 'Both passed; full submissions authorized'})
    for name in PIPELINES:
        full = submit_once(root, name + '-full', command(root, name + '-full', gpu=True, array=True,
                           minutes=plan['pipelines'][name]['full_minutes']))
        submit_once(root, name + '-score', command(root, name + '-score', dependency='afterok:' + full))
    atomic_json(root / 'full_submitted.json', {'both_arrays_submitted': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'preview', 'submit', 'gate'])
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.mode == 'prepare':
        prepare(root)
        return
    if args.mode == 'preview':
        for name in PIPELINES:
            for stage, minutes in [('validation', 60), ('full', 210 if name == 'legacy' else 294)]:
                cmd = command(root, name + '-' + stage, gpu=True, array=stage == 'full', minutes=minutes)
                result = subprocess.run(cmd[:1] + ['--test-only'] + cmd[1:], text=True, capture_output=True)
                print(shlex.join(cmd), '\n', result.stdout, result.stderr, flush=True)
                if result.returncode:
                    raise RuntimeError('Slurm rejected test-only request')
        return
    with open(root / 'submission.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if args.mode == 'gate':
            gate(root)
        else:
            jobs = [submit_once(root, name + '-validation', command(root, name + '-validation', gpu=True, minutes=60))
                    for name in PIPELINES]
            # afterany ensures a failure gets a recorded refusal, not a stranded dependency.
            submit_once(root, 'gate', command(root, 'gate', dependency='afterany:' + ':'.join(jobs)))


if __name__ == '__main__':
    main()
