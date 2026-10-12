"""Immutable prompt-rule campaigns; BCNB-only by default, smoke-gated arrays."""
import argparse
import datetime as dt
import difflib
import fcntl
import getpass
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, digest, load_config, read_json,
    read_manifest, resolve, sha256_file, write_manifest)
from data_processing.datasets import answer_label
from data_processing.sharding import make_shards
from eval.metrics import score

BASELINE = ROOT / 'runs/submissions/20260929_v3_bcnb1000_cptac_run'
ACCOUNT = 'cs6501-cbx8wm'
DATASETS = {'bcnb_1000_sample': {'count': 1000, 'shards': 10, 'smoke_minutes': 360,
    'minimum_full_minutes': 360, 'smoke_ids': (175, 879, 1403, 2728, 3519, 4647,
        4761, 5315, 5470, 6187, 6258, 6304, 6561)},
    'cptac': {'count': 240, 'shards': 6, 'smoke_minutes': 720,
    'minimum_full_minutes': 720, 'smoke_ids': (25, 38, 46, 56, 70, 78, 103, 105,
        108, 121, 134, 155, 199, 209, 220, 236)}}
TERMINAL_FAILURES = {'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL',
                     'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}


def now():
    return dt.datetime.now(ZoneInfo('America/New_York')).isoformat()


def log(message):
    print(now(), message, flush=True)


def run_command(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=90).stdout


def snapshot_info(source, config_path):
    code = ('import json; from data_processing.common import code_hash, load_config; '
            'from models.protocols import protocol_manifest; '
            'print(json.dumps({"source_hash":code_hash(),"prompts":'
            'protocol_manifest(load_config(' + repr(str(config_path)) + '))}))')
    result = subprocess.run([sys.executable, '-c', code], cwd=source, check=True,
                            capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def absolute_config(base, dataset):
    config = json.loads(json.dumps(base))
    for model in config['models'].values():
        model['path'] = str(resolve(model['path']))
    config['preprocessing']['segmentation_model']['path'] = str(resolve(
        config['preprocessing']['segmentation_model']['path']))
    for key in ('trident_path', 'trident_python'):
        config['preprocessing'][key] = str(resolve(config['preprocessing'][key]))
    for key in ('processed_root', 'manifest_root'):
        config['runtime'][key] = str(resolve(config['runtime'][key]))
    config['slurm'].update(account=ACCOUNT, shards=DATASETS[dataset]['shards'], max_concurrent_gpus=None)
    return config


def load_results(directory):
    results = {}
    for path in sorted(Path(directory).glob('shard_*/results/*.json')):
        row = read_json(path)
        if row['sample_id'] in results:
            raise ValueError('Duplicate result: ' + row['sample_id'])
        lock = read_json(path.parent.parent / 'run_lock.json')
        if row['run_hash'] != lock['run_hash']:
            raise ValueError('Result/lock identity mismatch: ' + str(path))
        results[row['sample_id']] = row
    return results


def historical_inputs(baseline, dataset, full):
    """Check existing inputs, outputs and non-resource scientific settings."""
    old = read_manifest(baseline / dataset / 'full.json')
    if old['samples_hash'] != full['samples_hash'] or old['samples'] != full['samples']:
        raise ValueError('Baseline question/image manifest differs: ' + dataset)
    results = load_results(baseline / dataset / 'full')
    metrics = score(full['samples'], results)
    saved_metrics = read_json(baseline / dataset / 'metrics.json')
    if not metrics['complete'] or metrics != saved_metrics:
        raise ValueError('Baseline metrics or completeness differ: ' + dataset)
    inventory, costs, configs = {}, {}, []
    for shard in sorted((baseline / dataset / 'full').glob('shard_*')):
        inv = read_json(shard / 'image_inventory.json')
        for image, value in inv.items():
            if image in inventory and inventory[image]['sha256'] != value['sha256']:
                raise ValueError('Conflicting baseline image hashes: ' + image)
            inventory[image] = value
        costs.update(read_json(shard / 'profile.json')['image_costs'])
        configs.append(read_json(shard / 'run_lock.json')['config'])
    expected_images = {row['image_id'] for row in full['samples']}
    if set(inventory) != expected_images or set(costs) != expected_images:
        raise ValueError('Incomplete baseline image/cost inventory')
    log(dataset + ': checking ' + str(len(inventory)) + ' image SHA256 values')
    def verify(item):
        image, value = item
        path = Path(value['path'])
        if sha256_file(path) != value['sha256']:
            raise ValueError('Original image changed: ' + image)
        return image, {'path': str(path), 'sha256': value['sha256'], 'bytes': path.stat().st_size,
                       'mtime_ns': path.stat().st_mtime_ns}
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = dict(pool.map(verify, inventory.items()))
    config = absolute_config(load_config('configs/reproduce_v3.yaml'), dataset)
    for previous in configs:
        for key in ('protocol', 'seed', 'models', 'algorithm', 'generation', 'preprocessing', 'runtime'):
            if config[key] != previous[key]:
                raise ValueError('Scientific baseline setting differs: ' + key)
    # Existing assets are content-locked; perform the same completeness/size checks as the runner.
    from scripts.run_inference import check_assets
    check_assets(config)
    env = read_json(next((baseline / dataset / 'full').glob('shard_*/environment.json')))
    current_packages = {}
    for name, old_version in env['packages'].items():
        try:
            current_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            current_packages[name] = None
        if current_packages[name] != old_version:
            raise ValueError('Baseline package version differs: ' + name)
    if str(Path(sys.executable).resolve()) != str(Path(env['executable']).resolve()):
        raise ValueError('Use the original pathagent Python environment')
    source_changes = {}
    for folder in ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval'):
        for file in (baseline / 'source' / folder).rglob('*.py'):
            relative = file.relative_to(baseline / 'source')
            current = ROOT / relative
            if file.read_bytes() != current.read_bytes():
                if str(relative) != 'data_processing/common.py':
                    raise ValueError('Unexpected non-prompt inference change: ' + str(relative))
                source_changes[str(relative)] = ''.join(difflib.unified_diff(
                    file.read_text().splitlines(True), current.read_text().splitlines(True)))
    if (baseline / 'source/pathagent.py').read_bytes() != (ROOT / 'pathagent.py').read_bytes():
        raise ValueError('Inference entrypoint changed')
    weighted_costs = {}
    for image, cost in costs.items():
        preparation = read_json(Path(inventory[image]['observation_cache']) / 'regions.json')['seconds']
        weighted_costs[image] = cost['seconds'] + preparation
    return {'metrics': metrics, 'images': verified, 'image_costs': weighted_costs,
            'packages': current_packages, 'allowed_resource_source_changes': source_changes,
            'results': results}


def job_script(source, config_path, manifest, output, source_hash, fleet, smoke=False, array=False):
    lines = ['#!/bin/bash', 'set -euo pipefail', 'cd ' + shlex.quote(str(source)),
             'export TOKENIZERS_PARALLELISM=false', 'export CUBLAS_WORKSPACE_CONFIG=:4096:8',
             'export TORCH_HOME=' + shlex.quote(str(ROOT / 'checkpoints/torch')),
             'export PATHAGENT_EXPECT_CODE_HASH=' + shlex.quote(source_hash),
             'export PATHAGENT_GPU_FLEET_LOCK=' + shlex.quote(str(fleet))]
    if array:
        lines += ['shard_index="${SLURM_ARRAY_TASK_ID:?Missing array index}"',
                  'manifest=' + shlex.quote(str(manifest)) + '/shard_${shard_index}.json',
                  'output=' + shlex.quote(str(output)) + '/shard_${shard_index}']
    else:
        lines += ['manifest=' + shlex.quote(str(manifest)), 'output=' + shlex.quote(str(output))]
    lines += ['exec ' + shlex.join([sys.executable, 'pathagent.py', '--config', str(config_path)])
              + ' --manifest "$manifest" --output "$output"' + (' --smoke' if smoke else '')]
    return '\n'.join(lines) + '\n'


def prepare(root, datasets=('bcnb_1000_sample',), baseline=BASELINE):
    import yaml
    from slurm.rule_prompts import VARIANTS, render_variant_sources
    if root.exists():
        raise ValueError('Use a new campaign directory')
    datasets = tuple(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - set(DATASETS):
        raise ValueError('Invalid enabled dataset list')
    base = load_config('configs/reproduce_v3.yaml')
    if base['seed'] != 128:
        raise ValueError('First-round ablations require seed=128')
    verified = {}
    for dataset in datasets:
        full = read_manifest(ROOT / 'data/manifests' / dataset / 'full.json')
        if len(full['samples']) != DATASETS[dataset]['count'] or not full.get('images_verified'):
            raise ValueError('Unexpected or unverified full manifest')
        if dataset == 'bcnb_1000_sample':
            frozen = read_json(ROOT / 'data/raw/bcnb_1000_sample/sampling.json')
            if [r['sample_id'] for r in full['samples']] != frozen['sample_ids']:
                raise ValueError('BCNB frozen sampling lock differs')
        verified[dataset] = historical_inputs(baseline, dataset, full)
        verified[dataset]['full'] = full
    root.mkdir(parents=True)
    plan = {'schema_version': 1, 'root': str(root), 'created_at': now(), 'seed': 128,
            'baseline_root': str(baseline), 'enabled_datasets': list(datasets), 'datasets': {},
            'variants': {}, 'account': ACCOUNT, 'qos': 'class', 'array_throttle': None,
            'gpu': 'RTX PRO 6000', 'system_submit_limit': 100, 'maximum_walltime_minutes': 4320,
            'baseline_is_reused': True, 'gate': 'Both enabled-dataset smoke jobs must pass for each variant',
            'files': {}}
    for dataset, history in verified.items():
        full = history.pop('full')
        results = history.pop('results')
        shared = root / 'shared_manifests' / dataset
        atomic_json(shared / 'full.json', full)
        prefix = 'bcnb' if dataset == 'bcnb_1000_sample' else dataset
        ids = {prefix + ':' + str(i) for i in DATASETS[dataset]['smoke_ids']}
        smoke_rows = [r for r in full['samples'] if r['sample_id'] in ids]
        if {r['sample_id'] for r in smoke_rows} != ids:
            raise ValueError('Smoke sample missing from frozen full manifest')
        write_manifest(shared / 'smoke.json', smoke_rows, dataset=full['dataset'], purpose='smoke',
                       images_verified=True, availability_restricted=False,
                       parent_samples_hash=full['samples_hash'], selection='fixed approved rule-audit smoke v0')
        count = DATASETS[dataset]['shards']
        shards, totals = make_shards(full['samples'], count, history['image_costs'])
        validate_shards(full['samples'], shards, count)
        for i, rows in enumerate(shards):
            write_manifest(shared / 'shards' / ('shard_' + str(i) + '.json'), rows,
                dataset=full['dataset'], purpose='shard', images_verified=True,
                availability_restricted=False, parent_samples_hash=full['samples_hash'],
                shard_index=i, shard_count=count)
        atomic_json(shared / 'baseline_compatibility.json', history)
        atomic_json(shared / 'sharding.json', {'method': 'LPT historical image inference + preparation',
                                             'image_costs': history['image_costs'], 'shard_seconds': totals})
        plan['datasets'][dataset] = {'full_manifest': str(shared / 'full.json'),
            'smoke_manifest': str(shared / 'smoke.json'), 'shard_manifests': str(shared / 'shards'),
            'samples_hash': full['samples_hash'], 'count': len(full['samples']), 'shards': count,
            'historical_shard_seconds': totals, 'baseline_metrics': history['metrics'],
            'baseline_smoke_agent_seconds': {r['sample_id']: results[r['sample_id']]['agent_seconds'] for r in smoke_rows}}
    files = [ROOT / 'pathagent.py']
    for folder in ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval', 'slurm'):
        files += list((ROOT / folder).rglob('*.py'))
    paper_source = (ROOT / 'models/paper_prompts.py').read_text()
    context_source = (ROOT / 'models/context_prompts.py').read_text()
    for variant in VARIANTS:
        base_dir = root / 'variants' / variant
        source = base_dir / 'source'
        for file in files:
            dest = source / file.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
        rendered = render_variant_sources(paper_source, context_source, variant)
        diffs = []
        for relative, text in rendered.items():
            previous = (source / relative).read_text()
            (source / relative).write_text(text)
            diffs += difflib.unified_diff(previous.splitlines(True), text.splitlines(True),
                                         fromfile='baseline/' + relative, tofile=variant + '/' + relative)
        (base_dir / 'prompt_diff.patch').write_text(''.join(diffs))
        record = {'root': str(base_dir), 'source': str(source), 'datasets': {}}
        for dataset in datasets:
            config = absolute_config(base, dataset)
            config_path = base_dir / 'configs' / (dataset + '.yaml')
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            load_config(config_path)
            info = snapshot_info(source, config_path)
            if 'source_hash' in record and record['source_hash'] != info['source_hash']:
                raise ValueError('Snapshot source identity changed during preparation')
            record.update(source_hash=info['source_hash'], prompt_manifest_hash=digest(info['prompts']))
            atomic_json(base_dir / 'prompt_manifest.json', info['prompts'])
            dataset_dir = base_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            shared = plan['datasets'][dataset]
            shutil.copy2(shared['full_manifest'], dataset_dir / 'full.json')
            shutil.copy2(shared['smoke_manifest'], dataset_dir / 'smoke.json')
            shutil.copytree(shared['shard_manifests'], dataset_dir / 'manifests')
            for stage in ('smoke', 'full'):
                manifest = Path(shared['smoke_manifest'] if stage == 'smoke' else shared['shard_manifests'])
                script = base_dir / (dataset + '-' + stage + '.sh')
                script.write_text(job_script(source, config_path, manifest, dataset_dir / stage,
                    info['source_hash'], root / 'gpu_fleet.json', smoke=stage == 'smoke', array=stage == 'full'))
            record['datasets'][dataset] = {'config': str(config_path), 'config_hash': digest(config)}
        plan['variants'][variant] = record
    # Controller also runs from a frozen copy, independent of subsequent workspace edits.
    coordinator = root / 'controller_source'
    for file in files:
        dest = coordinator / file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    for file in root.rglob('*'):
        if file.is_file():
            plan['files'][str(file)] = sha256_file(file)
    atomic_json(root / 'plan.json', plan)
    log('Prepared ' + str(root) + '; enabled datasets=' + ','.join(datasets) + '; variants=' + str(len(VARIANTS)))
    return plan


def validate_shards(samples, shards, count):
    if len(shards) != count or any(not shard for shard in shards):
        raise ValueError('Incorrect or empty shard count')
    owners, seen = {}, set()
    for index, rows in enumerate(shards):
        for row in rows:
            if row['sample_id'] in seen:
                raise ValueError('Duplicate sharded sample')
            seen.add(row['sample_id'])
            owner = owners.setdefault(row['image_id'], index)
            if owner != index:
                raise ValueError('One image split across shards')
    if seen != {row['sample_id'] for row in samples}:
        raise ValueError('Shard union differs from frozen full manifest')


def slurm_command(plan, variant, dataset, stage, minutes=None):
    if dataset not in plan['enabled_datasets']:
        raise ValueError('Dataset is not enabled in this campaign: ' + dataset)
    base = Path(plan['variants'][variant]['root'])
    minutes = minutes or DATASETS[dataset]['smoke_minutes']
    command = ['sbatch', '--parsable', '--account=' + plan['account'], '--qos=' + plan['qos'],
        '--partition=gpu', '--gres=gpu:rtx_pro_6000:1', '--constraint=rtxpro6000',
        '--cpus-per-task=8', '--mem=64G', '--time=' + str(minutes),
        '--job-name=rules-' + variant + '-' + dataset + '-' + stage,
        '--output=' + str(base / 'logs' / (dataset + '-' + stage + '-%A_%a.log'))]
    if stage == 'full':
        command.append('--array=0-' + str(plan['datasets'][dataset]['shards'] - 1))
    elif stage != 'smoke':
        raise ValueError('Invalid submission stage')
    return command + [str(base / (dataset + '-' + stage + '.sh'))]


def verify_frozen(plan):
    for path, expected in plan['files'].items():
        if sha256_file(path) != expected:
            raise ValueError('Frozen campaign file changed: ' + path)


def expected_run(plan, variant, dataset, manifest):
    v = plan['variants'][variant]
    config = load_config(v['datasets'][dataset]['config'])
    if digest(config) != v['datasets'][dataset]['config_hash']:
        raise ValueError('Frozen config identity differs')
    identity = {'config': config, 'code_hash': v['source_hash'], 'samples_hash': manifest['samples_hash'],
                'protocol': config['protocol'], 'prompt_manifest_hash': v['prompt_manifest_hash']}
    return identity, digest(identity)


def verify_smoke(plan, variant, dataset, job):
    out = Path(plan['variants'][variant]['root']) / dataset / 'smoke'
    manifest = read_manifest(plan['datasets'][dataset]['smoke_manifest'])
    identity, run_hash = expected_run(plan, variant, dataset, manifest)
    if read_json(out / 'run_lock.json') != {**identity, 'run_hash': run_hash}:
        raise ValueError('Smoke run-lock identity mismatch')
    profile = read_json(out / 'profile.json')
    required = {'passed': True, 'real_gpu': True, 'smoke': True, 'dataset': manifest['dataset'],
        'code_hash': identity['code_hash'], 'config_hash': digest(identity['config']),
        'manifest_hash': manifest['samples_hash'], 'run_hash': run_hash, 'slurm_job_id': str(job)}
    if any(profile.get(k) != value for k, value in required.items()):
        raise ValueError('Failed or stale real-GPU smoke profile')
    if not all(word in profile['gpu'].lower() for word in ('rtx', 'pro', '6000')):
        raise ValueError('Smoke used the wrong GPU')
    fleet = read_json(Path(plan['root']) / 'gpu_fleet.json')
    if any(profile.get(k) != v for k, v in fleet.items()):
        raise ValueError('GPU differs from campaign fleet')
    for row in manifest['samples']:
        result = read_json(out / 'results' / (digest(row['sample_id']) + '.json'))
        if (result.get('status') != 'ok' or result.get('sample_id') != row['sample_id']
                or result.get('run_hash') != run_hash
                or answer_label(result.get('pred_answer'), row['choices']) is None):
            raise ValueError('Invalid, missing or illegal smoke answer')
        for event in result.get('process', []):
            decision = event.get('decision', {})
            if str(decision.get('zoom_recommendation', '')).lower() == 'yes':
                allowed = identity['config']['algorithm']['zoom_magnifications']
                if type(decision.get('zoom_level')) is not int or decision['zoom_level'] not in allowed:
                    raise ValueError('Illegal smoke magnification')
    generation = read_json(out / 'resolved_generation.json')
    old = read_json(next((Path(plan['baseline_root']) / dataset / 'full').glob('shard_*/resolved_generation.json')))
    for key in ('executor', 'perceptor', 'overrides', 'image_processor', 'context_limit'):
        if generation[key] != old[key]:
            raise ValueError('Generation settings differ from baseline: ' + key)
    return profile


def full_walltime(plan, variant, dataset):
    base = Path(plan['variants'][variant]['root']) / dataset / 'smoke/results'
    manifest = read_manifest(plan['datasets'][dataset]['smoke_manifest'])
    grouped = defaultdict(lambda: [0.0, 0.0])
    for row in manifest['samples']:
        key = row.get('task') if dataset == 'bcnb_1000_sample' else row.get('source')
        grouped[key][0] += read_json(base / (digest(row['sample_id']) + '.json'))['agent_seconds']
        grouped[key][1] += plan['datasets'][dataset]['baseline_smoke_agent_seconds'][row['sample_id']]
    factor = max([1.0] + [new / max(old, 1.0) for new, old in grouped.values()])
    historical = max(plan['datasets'][dataset]['historical_shard_seconds'])
    minutes = max(DATASETS[dataset]['minimum_full_minutes'], math.ceil(2 * factor * historical / 60 + 30))
    if minutes > plan['maximum_walltime_minutes']:
        raise ValueError('Estimated full walltime exceeds partition limit; no sample truncation permitted')
    return minutes, {'smoke_qa_slowdown_factor': factor, 'historical_max_shard_seconds': historical,
                     'method': 'max task/source agent-time ratio; 2x historical inference+preparation +30min'}


def queue_rows():
    # The site's default view omits some of our pending jobs; explicit all
    # partitions/states is essential for both monitoring and submission limits.
    output = run_command(['squeue', '-a', '-t', 'all', '-h', '-r', '-u',
                          getpass.getuser(), '-o', '%i|%T|%a|%q'])
    return [line.strip().split('|') for line in output.splitlines() if line.strip()]


def job_state(job, queue, task_count=None):
    if any(row[0] == str(job) or row[0].startswith(str(job) + '_') for row in queue):
        return 'ACTIVE'
    text = run_command(['sacct', '-n', '-P', '-j', str(job), '--format=JobID%64,State,ExitCode'])
    records = {}
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) < 3 or '.' in fields[0]:
            continue
        job_id = fields[0].strip()
        if job_id == str(job) or job_id.startswith(str(job) + '_'):
            records[job_id] = (fields[1].split()[0].rstrip('+'), fields[2])
    if task_count is not None:
        expected = {str(job) + '_' + str(i) for i in range(task_count)}
        if not expected.issubset(records):
            return 'UNKNOWN'
        records = {key: value for key, value in records.items() if key in expected}
    if not records:
        return 'UNKNOWN'
    failures = [state for state, exitcode in records.values() if state in TERMINAL_FAILURES]
    if failures:
        return failures[0]
    if all(state == 'COMPLETED' and exitcode == '0:0' for state, exitcode in records.values()):
        return 'COMPLETED'
    if any(state == 'COMPLETED' and exitcode != '0:0' for state, exitcode in records.values()):
        return 'FAILED'
    return 'UNKNOWN'


def submit_once(plan, variant, dataset, stage, queue, minutes=None):
    base = Path(plan['variants'][variant]['root'])
    receipt = base / 'receipts' / (dataset + '-' + stage + '.json')
    if receipt.exists():
        return read_json(receipt)['job_id']
    occupied = sum(row[3] == plan['qos'] for row in queue)
    needed = plan['datasets'][dataset]['shards'] if stage == 'full' else 1
    if occupied + needed > plan['system_submit_limit']:
        return None
    command = slurm_command(plan, variant, dataset, stage, minutes)
    (base / 'logs').mkdir(exist_ok=True)
    try:
        response = run_command(command)
    except subprocess.CalledProcessError as exc:
        if any(word in (exc.stderr or '').lower() for word in ('qosmaxsubmit', 'job violates accounting/qos', 'job submit limit')):
            log('System submission capacity unavailable; waiting: ' + variant + '/' + stage)
            return None
        raise
    job = response.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Slurm did not return a numeric job ID: ' + response)
    atomic_json(receipt, {'job_id': job, 'command': command, 'submitted_at': now()})
    verification = run_command(['scontrol', 'show', 'job', job, '-o'])
    atomic_json(base / 'receipts' / (dataset + '-' + stage + '-verification.json'), {'job_id': job, 'verification': verification})
    log('Accepted ' + variant + '/' + dataset + '/' + stage + ' job=' + job)
    return job


def verify_full(plan, variant, dataset):
    base = Path(plan['variants'][variant]['root']) / dataset
    manifest = read_manifest(plan['datasets'][dataset]['full_manifest'])
    results = load_results(base / 'full')
    for index in range(plan['datasets'][dataset]['shards']):
        shard_manifest = read_manifest(Path(plan['datasets'][dataset]['shard_manifests']) / ('shard_' + str(index) + '.json'))
        identity, run_hash = expected_run(plan, variant, dataset, shard_manifest)
        shard = base / 'full' / ('shard_' + str(index))
        if read_json(shard / 'run_lock.json') != {**identity, 'run_hash': run_hash}:
            raise ValueError('Full shard identity mismatch')
        if any(results.get(row['sample_id'], {}).get('run_hash') != run_hash for row in shard_manifest['samples']):
            raise ValueError('Full shard result missing or has wrong identity')
    metrics = score(manifest['samples'], results)
    atomic_json(base / 'metrics.json', metrics)
    if not metrics['complete']:
        raise ValueError('Incomplete full run')
    return metrics


def controller(root, poll_seconds=30):
    plan = read_json(root / 'plan.json')
    with open(root / 'controller.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_frozen(plan)
        state_path = root / 'status.json'
        state = read_json(state_path) if state_path.exists() else {'created_at': now(), 'variants': {}}
        state.update(controller_pid=os.getpid(), controller_started_at=now(), enabled_datasets=plan['enabled_datasets'])
        last_summary = None
        while True:
            queue = queue_rows()
            for variant, record in plan['variants'].items():
                status = state['variants'].setdefault(variant, {'stage': 'smoke_pending', 'jobs': {}})
                if status['stage'] in ('complete', 'smoke_failed', 'full_failed'):
                    continue
                try:
                    if status['stage'].startswith('smoke'):
                        for dataset in plan['enabled_datasets']:
                            job = submit_once(plan, variant, dataset, 'smoke', queue)
                            if job:
                                status['jobs'][dataset + '-smoke'] = job
                                queue = queue_rows()
                        if len(status['jobs']) < len(plan['enabled_datasets']):
                            continue
                        states = {d: job_state(status['jobs'][d + '-smoke'], queue) for d in plan['enabled_datasets']}
                        status.update(stage='smoke_running', smoke_job_states=states)
                        if any(value in TERMINAL_FAILURES for value in states.values()):
                            status.update(stage='smoke_failed', error='Slurm smoke failure: ' + str(states))
                            log(variant + ': smoke failed; full not submitted')
                        elif all(value == 'COMPLETED' for value in states.values()):
                            verify_frozen(plan)
                            profiles = {d: verify_smoke(plan, variant, d, status['jobs'][d + '-smoke']) for d in plan['enabled_datasets']}
                            atomic_json(Path(record['root']) / 'gate_decision.json',
                                {'passed': True, 'checked_at': now(), 'profiles': profiles, 'datasets': plan['enabled_datasets']})
                            status['stage'] = 'full_pending'
                            log(variant + ': smoke passed; releasing full arrays')
                    if status['stage'].startswith('full'):
                        for dataset in plan['enabled_datasets']:
                            minutes, estimate = full_walltime(plan, variant, dataset)
                            atomic_json(Path(record['root']) / dataset / 'walltime_estimate.json', {'minutes': minutes, **estimate})
                            job = submit_once(plan, variant, dataset, 'full', queue, minutes)
                            if job:
                                status['jobs'][dataset + '-full'] = job
                                queue = queue_rows()
                        if any(d + '-full' not in status['jobs'] for d in plan['enabled_datasets']):
                            continue
                        states = {d: job_state(status['jobs'][d + '-full'], queue, plan['datasets'][d]['shards'])
                                  for d in plan['enabled_datasets']}
                        status.update(stage='full_running', full_job_states=states)
                        if any(value in TERMINAL_FAILURES for value in states.values()):
                            status.update(stage='full_failed', error='Slurm full failure: ' + str(states))
                            log(variant + ': full array failed; retaining all partial results')
                        elif all(value == 'COMPLETED' for value in states.values()):
                            status['metrics'] = {d: verify_full(plan, variant, d) for d in plan['enabled_datasets']}
                            status['stage'] = 'complete'
                            log(variant + ': complete')
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    failed_stage = 'full_failed' if status['stage'].startswith('full') else 'smoke_failed'
                    status.update(stage=failed_stage, error=str(exc))
                    atomic_json(Path(record['root']) / 'gate_decision.json', {'passed': False,
                                'checked_at': now(), 'reason': str(exc),
                                'full_submitted': any(key.endswith('-full') for key in status['jobs'])})
                    log(variant + ': ' + failed_stage + ': ' + str(exc))
            state['updated_at'] = now()
            state['system_queue_tasks'] = len(queue)
            atomic_json(state_path, state)
            summary = {v: value['stage'] for v, value in state['variants'].items()}
            if summary != last_summary:
                log('Campaign state: ' + json.dumps(summary))
                last_summary = summary
            if all(value in ('complete', 'smoke_failed', 'full_failed') for value in summary.values()):
                report = root / 'controller_source/slurm/report_rule_ablations.py'
                if report.exists():
                    subprocess.run([sys.executable, str(report), '--root', str(root)], check=True)
                state.update(finished_at=now(), complete=all(value == 'complete' for value in summary.values()))
                atomic_json(state_path, state)
                log('Campaign finished; complete=' + str(state['complete']))
                return state
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'preview', 'run', 'status', 'report'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS), default=['bcnb_1000_sample'])
    parser.add_argument('--baseline', default=str(BASELINE))
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.mode == 'prepare':
        prepare(root, args.datasets, Path(args.baseline).resolve())
    elif args.mode == 'run':
        if not 10 <= args.poll_seconds <= 60:
            parser.error('poll-seconds must be between 10 and 60')
        controller(root, args.poll_seconds)
    elif args.mode == 'status':
        print(json.dumps(read_json(root / 'status.json'), indent=2))
    elif args.mode == 'preview':
        plan = read_json(root / 'plan.json')
        verify_frozen(plan)
        for variant in plan['variants']:
            for dataset in plan['enabled_datasets']:
                for stage in ('smoke', 'full'):
                    minutes = DATASETS[dataset]['smoke_minutes'] if stage == 'smoke' else DATASETS[dataset]['minimum_full_minutes']
                    cmd = slurm_command(plan, variant, dataset, stage, minutes)
                    preview = run_command(cmd[:1] + ['--test-only'] + cmd[1:])
                    log(variant + '/' + dataset + '/' + stage + ': ' + preview.strip())
    else:
        subprocess.run([sys.executable, str(root / 'controller_source/slurm/report_rule_ablations.py'),
                        '--root', str(root)], check=True)


if __name__ == '__main__':
    main()
