"""Frozen v3 validation and gated BCNB-1000/CPTAC arrays; no smoke waiver."""
import argparse
import fcntl
import math
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, code_hash, digest, load_config,
    read_json, read_manifest, resolve, run_identity, sha256_file, write_manifest)
from data_processing.regions import ImageSource
from data_processing.sharding import make_shards
from models.protocols import CONTEXT_PROTOCOL, protocol_name
from scripts.run_inference import check_assets
from scripts.submit_runs import script_text
from slurm.submit_bcnb_comparison import command, submit_once, shell, completed_successfully, ACCOUNT

DATASETS = ('bcnb_1000_sample', 'cptac')


def prepare(root):
    import yaml
    if root.exists():
        raise ValueError('Use a new submission directory')
    config = load_config('configs/reproduce_v3.yaml')
    if protocol_name(config) != CONTEXT_PROTOCOL:
        raise ValueError('Expected v3 protocol')
    config['slurm']['account'] = ACCOUNT
    for model in config['models'].values():
        model['path'] = str(resolve(model['path']))
    config['preprocessing']['segmentation_model']['path'] = str(resolve(config['preprocessing']['segmentation_model']['path']))
    for key in ('trident_path', 'trident_python'):
        config['preprocessing'][key] = str(resolve(config['preprocessing'][key]))
    for key in ('processed_root', 'manifest_root'):
        config['runtime'][key] = str(resolve(config['runtime'][key]))
    check_assets(config)
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    config_path = root / 'reproduce_v3.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = {'root': str(root), 'source': str(root / 'source'), 'source_hash': code_hash(),
            'config': str(config_path), 'config_hash': digest(config), 'datasets': {},
            'gate_rule': 'Both real GPU validations must pass before either benchmark is submitted',
            'account': ACCOUNT, 'qos': 'class', 'shards_per_dataset': 16,
            'gpu': 'RTX PRO 6000', 'files': {}}
    for dataset in DATASETS:
        full = read_manifest(f'data/manifests/{dataset}/full.json')
        if not full.get('images_verified') or full.get('availability_restricted'):
            raise ValueError('Complete image-verified manifest required')
        if dataset == 'bcnb_1000_sample':
            frozen = read_json(resolve('data/raw/bcnb_1000_sample/sampling.json'))
            if [r['sample_id'] for r in full['samples']] != frozen['sample_ids']:
                raise ValueError('BCNB frozen question IDs changed')
        if len(full['samples']) != (1000 if dataset == 'bcnb_1000_sample' else 240):
            raise ValueError('Unexpected question count')
        base = root / dataset
        atomic_json(base / 'full.json', full)
        groups = defaultdict(list)
        for row in full['samples']:
            groups[row['image_id']].append(row)
        geometry, costs, cohorts = {}, {}, defaultdict(list)
        for image, rows in groups.items():
            with ImageSource(rows[0]['image_path'], rows[0]['assumed_mpp']) as src:
                if src.mpp is None:
                    raise ValueError('Missing physical scale: ' + image)
                geometry[image] = {'width': src.width, 'height': src.height,
                                   'mpp': src.mpp, 'mpp_source': src.mpp_source}
                tiles = math.ceil(src.width / 4096) * math.ceil(src.height / 4096)
                costs[image] = tiles * 30 + len(rows) * 180
                cohorts[rows[0]['source']].append((src.width * src.height, image))
        smoke_ids = set()
        for cohort in cohorts.values():
            cohort.sort()
            for i in (0, len(cohort) // 2, len(cohort) - 1):
                smoke_ids.update(r['sample_id'] for r in groups[cohort[i][1]])
        # Ensure all BCNB task types occur, without selecting using the gold answer.
        for task in sorted({r['task'] for r in full['samples']}):
            task_rows = [r for r in full['samples'] if r['task'] == task]
            if not any(r['sample_id'] in smoke_ids for r in task_rows):
                smoke_ids.add(min(task_rows, key=lambda r: digest(r['sample_id']))['sample_id'])
        smoke = [r for r in full['samples'] if r['sample_id'] in smoke_ids]
        write_manifest(base / 'smoke.json', smoke, dataset=full['dataset'], purpose='smoke',
            images_verified=True, availability_restricted=False, parent_samples_hash=full['samples_hash'],
            selection='smallest/median/largest original by source, plus task coverage')
        shards, totals = make_shards(full['samples'], 16, costs)
        for i, rows in enumerate(shards):
            write_manifest(base / 'manifests' / f'shard_{i}.json', rows, dataset=full['dataset'],
                purpose='shard', images_verified=True, availability_restricted=False,
                parent_samples_hash=full['samples_hash'], shard_index=i, shard_count=16)
        atomic_json(base / 'geometry.json', geometry)
        atomic_json(base / 'sharding.json', {'method': 'LPT grid upper bound; 30s/region + 180s/question engineering prior',
            'image_costs': costs, 'shard_costs': totals, 'measured': False})
        plan['datasets'][dataset] = {'questions': len(full['samples']), 'images': len(groups),
            'validation_questions': len(smoke), 'validation_images': len({r['image_id'] for r in smoke}),
            'full_minutes': max(240, min(1440, math.ceil(max(totals) * 1.5 / 60 + 30))),
            'validation_minutes': 180 if dataset == 'bcnb_1000_sample' else 720}
        for stage in ('validation', 'full'):
            text = script_text(base / ('smoke.json' if stage == 'validation' else 'manifests'),
                base / stage, config_path, plan['source_hash'], smoke=stage == 'validation',
                array=stage == 'full', fleet=root / 'gpu_fleet.json')
            text = text.replace('cd ' + shlex.quote(str(ROOT)), 'cd ' + shlex.quote(plan['source']), 1)
            text = text.replace('export TOKENIZERS_PARALLELISM=false',
                'export TORCH_HOME=' + shlex.quote(str(ROOT / 'checkpoints/torch')) + '\nexport TOKENIZERS_PARALLELISM=false', 1)
            (root / (dataset + '-' + stage + '.sh')).write_text(text)
        (root / (dataset + '-score.sh')).write_text(shell(Path(plan['source']), ['scripts/evaluate.py',
            '--manifest', str(base / 'full.json'), '--results-dir', str(base / 'full'),
            '--output', str(base / 'metrics.json')]))
    files = [ROOT / 'pathagent.py', Path(__file__).resolve(), ROOT / 'slurm/submit_bcnb_comparison.py']
    for folder in ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval'):
        files.extend((ROOT / folder).rglob('*.py'))
    for file in files:
        dest = Path(plan['source']) / file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    (root / 'gate.sh').write_text(shell(Path(plan['source']), ['slurm/submit_v3_benchmarks.py', 'gate', '--root', str(root)]))
    for file in root.rglob('*'):
        if file.is_file():
            plan['files'][str(file)] = sha256_file(file)
    atomic_json(root / 'plan.json', plan)
    print({'root': str(root), 'datasets': plan['datasets']}, flush=True)


def verify_validation(plan, dataset, job):
    root = Path(plan['root'])
    config = load_config(plan['config'])
    if digest(config) != plan['config_hash']:
        raise RuntimeError('Config changed')
    manifest = read_manifest(root / dataset / 'smoke.json')
    out = root / dataset / 'validation'
    identity = run_identity(config, plan['source_hash'], manifest['samples_hash'])
    run_hash = digest(identity)
    if read_json(out / 'run_lock.json') != {**identity, 'run_hash': run_hash}:
        raise RuntimeError('Validation identity mismatch')
    profile = read_json(out / 'profile.json')
    required = dict(passed=True, real_gpu=True, smoke=True, dataset=manifest['dataset'],
        code_hash=plan['source_hash'], config_hash=plan['config_hash'],
        manifest_hash=manifest['samples_hash'], run_hash=run_hash, slurm_job_id=str(job))
    if any(profile.get(k) != v for k, v in required.items()):
        raise RuntimeError('Missing or stale validation: ' + dataset)
    fleet = read_json(root / 'gpu_fleet.json')
    if any(profile.get(k) != v for k, v in fleet.items()):
        raise RuntimeError('Validation GPU differs from fleet')
    if not all(s in profile['gpu'].lower() for s in ('rtx', 'pro', '6000')):
        raise RuntimeError('Wrong GPU model')
    for row in manifest['samples']:
        result = read_json(out / 'results' / (digest(row['sample_id']) + '.json'))
        if (result.get('status') != 'ok' or result.get('sample_id') != row['sample_id']
                or result.get('run_hash') != run_hash):
            raise RuntimeError('Invalid or missing validation result')
    return profile


def gate(root):
    plan = read_json(root / 'plan.json')
    try:
        if code_hash() != plan['source_hash']:
            raise RuntimeError('Source changed')
        for path, checksum in plan['files'].items():
            if sha256_file(path) != checksum:
                raise RuntimeError('Frozen file changed: ' + path)
        jobs = {d: read_json(root / 'receipts' / (d + '-validation.json'))['job_id'] for d in DATASETS}
        completed_successfully(list(jobs.values()))
        profiles = {d: verify_validation(plan, d, jobs[d]) for d in DATASETS}
    except Exception as exc:
        atomic_json(root / 'gate_decision.json', {'passed': False, 'reason': str(exc), 'full_submitted': False})
        raise
    atomic_json(root / 'gate_decision.json', {'passed': True, 'validation_jobs': jobs, 'profiles': profiles})
    for d in DATASETS:
        full = submit_once(root, d + '-full', command(root, d + '-full', gpu=True, array=True,
            minutes=plan['datasets'][d]['full_minutes']))
        submit_once(root, d + '-score', command(root, d + '-score', dependency='afterok:' + full, minutes=15))
    atomic_json(root / 'full_submitted.json', {'both_arrays_submitted': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'preview', 'submit', 'gate'])
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.mode == 'prepare':
        prepare(root)
        return
    plan = read_json(root / 'plan.json')
    if args.mode == 'preview':
        for d in DATASETS:
            for stage in ('validation', 'full'):
                cmd = command(root, d + '-' + stage, gpu=True, array=stage == 'full',
                    minutes=plan['datasets'][d][stage + '_minutes'])
                result = subprocess.run(cmd[:1] + ['--test-only'] + cmd[1:], text=True, capture_output=True)
                print(result.stdout, result.stderr, flush=True)
                result.check_returncode()
        return
    with open(root / 'submission.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if args.mode == 'gate':
            gate(root)
        else:
            jobs = [submit_once(root, d + '-validation', command(root, d + '-validation', gpu=True,
                minutes=plan['datasets'][d]['validation_minutes'])) for d in DATASETS]
            submit_once(root, 'gate', command(root, 'gate', dependency='afterany:' + ':'.join(jobs)))


if __name__ == '__main__':
    main()
