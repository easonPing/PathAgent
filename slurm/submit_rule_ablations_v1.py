"""BCNB-only frozen A/B/C/D study with a shared validation gate."""
import argparse
import difflib
import fcntl
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, digest, load_config, read_json,
    read_manifest, resolve, sha256_file, write_manifest)
from data_processing.sharding import make_shards
from eval.metrics import score
from models.protocols import protocol_manifest
from slurm.rule_prompts_v1 import CONTROL, VARIANTS, render_variant_sources, selected_packages, zoom_enabled
from slurm import submit_rule_ablations as old

DATASET = 'bcnb_1000_sample'
VALIDATION_IDS = (336, 879, 1090, 2286, 3601, 4573, 4879, 5329, 7071, 7271)
ALLOWED_SOURCE_CHANGES = {'data_processing/common.py', 'data_processing/preprocess.py', 'models/agent.py'}
PERCEPTOR_TEMPLATES = ('PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'QUESTION_PROMPT', 'PERCEPTOR_USER', 'ZOOM_MISSING_USER')
EVENT_ADDITIONS = {'current_parent_regions', 'active_zoom_regions', 'requested_action',
                  'actual_action', 'action_executed', 'zoom_unvisited_candidate_count'}


def stable_trace(result):
    """All scientific inputs/outputs, excluding timing and added bookkeeping."""
    process = [{k: v for k, v in event.items() if k not in EVENT_ADDITIONS}
               for event in result.get('process', [])]
    calls = [{k: v for k, v in call.items() if k != 'seconds'}
             for call in result.get('model_calls', [])]
    return {key: result.get(key) for key in ('pred_answer', 'explanation', 'status', 'stop_reason',
            'final_response', 'initial_region_count', 'total_examined_regions')} | {
                'process': process, 'model_calls': calls}


def historical_inputs(baseline, full):
    previous = read_manifest(baseline / DATASET / 'full.json')
    if previous['samples'] != full['samples'] or previous['samples_hash'] != full['samples_hash']:
        raise ValueError('BCNB baseline manifest differs')
    results = old.load_results(baseline / DATASET / 'full')
    metrics = score(full['samples'], results)
    if not metrics['complete'] or metrics != read_json(baseline / DATASET / 'metrics.json'):
        raise ValueError('Baseline result completeness or metrics differ')
    shards = sorted((baseline / DATASET / 'full').glob('shard_*'))
    base = read_json(shards[0] / 'run_lock.json')['config']
    current = old.absolute_config(load_config('configs/reproduce_v3.yaml'), DATASET)
    for key in ('protocol', 'seed', 'models', 'algorithm', 'generation', 'preprocessing', 'runtime'):
        if current[key] != base[key]:
            raise ValueError('Scientific setting differs from baseline: ' + key)
    inventory, costs, fleet = {}, {}, None
    for shard in shards:
        lock = read_json(shard / 'run_lock.json')
        for key in ('protocol', 'seed', 'models', 'algorithm', 'generation', 'preprocessing', 'runtime'):
            if lock['config'][key] != base[key]:
                raise ValueError('Baseline shards have incompatible scientific settings')
        for image, record in read_json(shard / 'image_inventory.json').items():
            if image in inventory and inventory[image] != record:
                raise ValueError('Incompatible baseline image cache')
            inventory[image] = record
        profile = read_json(shard / 'profile.json')
        spec = {key: profile[key] for key in ('gpu', 'gpu_memory_bytes')}
        if fleet is not None and spec != fleet:
            raise ValueError('Baseline GPU fleet differs across shards')
        fleet = spec
        costs.update(profile['image_costs'])
    expected = {row['image_id'] for row in full['samples']}
    if set(inventory) != expected or set(costs) != expected:
        raise ValueError('Baseline image inventory incomplete')
    old.log('Checking original SHA256 for ' + str(len(expected)) + ' BCNB images')
    def verify(item):
        image, record = item
        path = Path(record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('Original image changed: ' + image)
        return image, {**record, 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = dict(pool.map(verify, inventory.items()))
    from scripts.run_inference import check_assets
    check_assets(base)
    env = read_json(shards[0] / 'environment.json')
    if Path(sys.executable).resolve() != Path(env['executable']).resolve():
        raise ValueError('Use the original pathagent Python')
    for name, version in env['packages'].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            raise ValueError('Environment differs: ' + name)
    changes = {}
    for folder in ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval'):
        frozen = {str(p.relative_to(baseline / 'source')): p for p in (baseline / 'source' / folder).rglob('*.py')}
        workspace = {str(p.relative_to(ROOT)): p for p in (ROOT / folder).rglob('*.py')}
        if set(frozen) != set(workspace):
            raise ValueError('Unexpected inference source file set change: ' + folder)
        for relative, previous_path in frozen.items():
            current_path = workspace[relative]
            if previous_path.read_bytes() != current_path.read_bytes():
                if relative not in ALLOWED_SOURCE_CHANGES:
                    raise ValueError('Unplanned inference source change: ' + relative)
                changes[relative] = ''.join(difflib.unified_diff(previous_path.read_text().splitlines(True),
                    current_path.read_text().splitlines(True), fromfile='baseline/' + relative, tofile=relative))
    if (baseline / 'source/pathagent.py').read_bytes() != (ROOT / 'pathagent.py').read_bytes():
        raise ValueError('Inference entrypoint changed')
    weighted = {image: cost['seconds'] + read_json(Path(inventory[image]['observation_cache']) /
                'regions.json')['seconds'] for image, cost in costs.items()}
    return base, results, {'metrics': metrics, 'images': inventory, 'image_costs': weighted,
                          'packages': env['packages'], 'gpu_fleet': fleet, 'planned_source_changes': changes}


def freeze_initial_inputs(root, baseline, config, history, full):
    prompts = protocol_manifest(config)
    observation_key = digest({'models': config['models'], 'generation': config['generation'],
                             'seed': config['seed'], 'protocol': prompts})
    manifest = {'schema_version': 1, 'kind': 'frozen_initial_inputs', 'source_baseline': str(baseline),
        'samples_hash': full['samples_hash'], 'contract': {key: config[key] for key in
            ('seed', 'models', 'generation', 'preprocessing')}, 'images': {}}
    manifest['contract'].update(initial_magnification=config['algorithm']['initial_magnification'],
        perceptor_prompt_hashes={name: prompts['templates'][name]['sha256'] for name in PERCEPTOR_TEMPLATES})
    counts = {'images': 0, 'regions': 0, 'generic_model_calls_copied': 0}
    samples = {}
    for sample in full['samples']:
        samples.setdefault(sample['image_id'], sample)
    baseline_source_hash = read_json(next((baseline / DATASET / 'full').glob('shard_*/run_lock.json')))['code_hash']
    for image, inventory in sorted(history['images'].items()):
        source = Path(inventory['observation_cache'])
        regions = read_json(source / 'regions.json')
        expected_key = digest({'image': inventory['sha256'], 'image_path': str(resolve(inventory['path'])),
            'kind': samples[image]['input_kind'], 'mpp': samples[image]['assumed_mpp'], 'preprocessing': config['preprocessing'],
            'initial_magnification': config['algorithm']['initial_magnification'],
            'code': baseline_source_hash})
        if regions['key'] != expected_key or regions['image_sha256'] != inventory['sha256']:
            raise ValueError('Historical region cache identity differs: ' + image)
        observations_path = source / ('observations_' + observation_key + '.json')
        observations = read_json(observations_path)
        names = {region['region_id'] for region in regions['regions']}
        if names != set(observations['descriptions']) or names != set(observations['features']):
            raise ValueError('Historical observation coverage differs')
        dest = root / 'shared_initial_inputs' / image
        dest.mkdir(parents=True)
        shutil.copy2(source / 'regions.json', dest / 'regions.json')
        shutil.copy2(observations_path, dest / 'observations.json')
        # Preserve full Generic inference provenance without fresh inference or writable links.
        for name in sorted(names):
            partial = source / ('description_' + digest([observation_key, name]) + '.json')
            description = read_json(partial)
            if description['description'] != observations['descriptions'][name]:
                raise ValueError('Historical Generic description differs: ' + image + '/' + name)
            shutil.copy2(partial, dest / partial.name)
            counts['generic_model_calls_copied'] += 1
        canonical = str(Path(inventory['path']).resolve())
        if any(region['image_path'] != canonical for region in regions['regions']):
            raise ValueError('Historical canonical region path differs')
        provenance = {'source_cache': str(source), 'source_observations': str(observations_path),
            'observation_key': observation_key, 'baseline': str(baseline),
            'generic_fresh_calls': 0, 'region_count': len(names)}
        atomic_json(dest / 'source_provenance.json', provenance)
        manifest['images'][image] = {'directory': str(dest), 'image_path': canonical,
            'image_sha256': inventory['sha256'], 'width': inventory['width'], 'height': inventory['height'],
            'regions_sha256': sha256_file(dest / 'regions.json'),
            'observations_sha256': sha256_file(dest / 'observations.json'), **provenance}
        manifest['images'][image].update({key: samples[image][key] for key in ('dataset', 'input_kind', 'assumed_mpp')})
        counts['images'] += 1
        counts['regions'] += len(names)
    path = root / 'initial_inputs.json'
    atomic_json(path, manifest)
    return {'manifest': str(path), 'sha256': sha256_file(path), **counts,
            'generic_fresh_calls': 0, 'source': 'frozen baseline initial regions, Generic descriptions and PLIP embeddings'}


def prepare(root, baseline=old.BASELINE):
    import yaml
    if root.exists():
        raise ValueError('Use a new campaign directory')
    full = read_manifest(ROOT / 'data/manifests' / DATASET / 'full.json')
    sampling = read_json(ROOT / 'data/raw/bcnb_1000_sample/sampling.json')
    if (len(full['samples']) != 1000 or not full.get('images_verified')
            or [r['sample_id'] for r in full['samples']] != sampling['sample_ids']):
        raise ValueError('BCNB1000 sampling lock differs')
    base, results, history = historical_inputs(baseline, full)
    root.mkdir(parents=True)
    atomic_json(root / 'gpu_fleet.json', history['gpu_fleet'])
    shared = root / 'shared_manifests' / DATASET
    atomic_json(shared / 'full.json', full)
    ids = {'bcnb:' + str(i) for i in VALIDATION_IDS}
    validation = [row for row in full['samples'] if row['sample_id'] in ids]
    if {row['sample_id'] for row in validation} != ids:
        raise ValueError('Approved validation sample missing')
    write_manifest(shared / 'validation.json', validation, dataset=full['dataset'], purpose='smoke',
        images_verified=True, availability_restricted=False, parent_samples_hash=full['samples_hash'],
        selection='fixed approved v1 validation; technical gate only')
    shards, totals = make_shards(full['samples'], 10, history['image_costs'])
    old.validate_shards(full['samples'], shards, 10)
    for i, rows in enumerate(shards):
        write_manifest(shared / 'shards' / ('shard_' + str(i) + '.json'), rows,
            dataset=full['dataset'], purpose='shard', images_verified=True, availability_restricted=False,
            parent_samples_hash=full['samples_hash'], shard_index=i, shard_count=10)
    atomic_json(shared / 'baseline_compatibility.json', history)
    atomic_json(shared / 'baseline_validation_results.json', {key: results[key] for key in sorted(ids)})
    atomic_json(shared / 'sharding.json', {'method': 'LPT historical image inference + preparation',
                                         'image_costs': history['image_costs'], 'shard_seconds': totals})
    initial = freeze_initial_inputs(root, baseline, base, history, full)
    plan = {'schema_version': 1, 'rule_version': 'rules-v1', 'root': str(root), 'created_at': old.now(),
        'seed': base['seed'], 'baseline_root': str(baseline), 'baseline_is_reused': True,
        'enabled_datasets': [DATASET], 'datasets': {DATASET: {
            'full_manifest': str(shared / 'full.json'), 'validation_manifest': str(shared / 'validation.json'),
            'smoke_manifest': str(shared / 'validation.json'), 'shard_manifests': str(shared / 'shards'),
            'samples_hash': full['samples_hash'], 'count': 1000, 'shards': 10,
            'historical_shard_seconds': totals, 'baseline_metrics': history['metrics'],
            'baseline_smoke_agent_seconds': {row['sample_id']: results[row['sample_id']]['agent_seconds'] for row in validation}}},
        'initial_inputs': initial, 'variants': {}, 'controls': {}, 'account': old.ACCOUNT, 'qos': 'class',
        'gpu': 'RTX PRO 6000', 'array_throttle': None, 'system_submit_limit': 100,
        'maximum_walltime_minutes': 4320, 'gate': 'All six technical validations pass before any full array',
        'invariants': {'zoom_review_arm': 'C_ZOOM_LOOP', 'baseline_no_zoom_expected': 954}, 'files': {}}
    files = [ROOT / 'pathagent.py']
    for folder in ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval', 'slurm'):
        files.extend((ROOT / folder).rglob('*.py'))
    paper = (ROOT / 'models/paper_prompts.py').read_text()
    context = (ROOT / 'models/context_prompts.py').read_text()
    for variant in (CONTROL,) + VARIANTS:
        directory = root / ('controls' if variant == CONTROL else 'variants') / variant
        source = directory / 'source'
        for path in files:
            dest = source / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        diffs = []
        for relative, rendered in render_variant_sources(paper, context, variant).items():
            previous = (source / relative).read_text()
            (source / relative).write_text(rendered)
            diffs.extend(difflib.unified_diff(previous.splitlines(True), rendered.splitlines(True),
                fromfile='baseline/' + relative, tofile=variant + '/' + relative))
        (directory / 'prompt_diff.patch').write_text(''.join(diffs))
        config = old.absolute_config(base, DATASET)
        config['algorithm']['zoom_return_to_loop'] = zoom_enabled(variant)
        config['runtime'].update(initial_inputs_manifest=initial['manifest'], initial_inputs_hash=initial['sha256'])
        config['runtime']['processed_root'] = str(root / 'unused_legacy_cache' / variant)
        config_path = directory / 'configs' / (DATASET + '.yaml')
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_config(config_path)
        info = old.snapshot_info(source, config_path)
        atomic_json(directory / 'prompt_manifest.json', info['prompts'])
        out = directory / DATASET
        out.mkdir()
        shutil.copy2(shared / 'full.json', out / 'full.json')
        shutil.copy2(shared / 'validation.json', out / 'validation.json')
        shutil.copytree(shared / 'shards', out / 'manifests')
        stages = ('validation',) if variant == CONTROL else ('validation', 'full')
        for stage in stages:
            manifest = shared / ('validation.json' if stage == 'validation' else 'shards')
            (directory / (DATASET + '-' + stage + '.sh')).write_text(old.job_script(source, config_path,
                manifest, out / stage, info['source_hash'], root / 'gpu_fleet.json',
                smoke=stage == 'validation', array=stage == 'full'))
        record = {'root': str(directory), 'source': str(source), 'source_hash': info['source_hash'],
            'prompt_manifest_hash': digest(info['prompts']), 'rules': list(selected_packages(variant)),
            'post_zoom_reflect': zoom_enabled(variant),
            'datasets': {DATASET: {'config': str(config_path), 'config_hash': digest(config)}}}
        plan['controls' if variant == CONTROL else 'variants'][variant] = record
    for path in files:
        dest = root / 'controller_source' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for path in root.rglob('*'):
        if path.is_file():
            plan['files'][str(path)] = sha256_file(path)
    atomic_json(root / 'plan.json', plan)
    old.log('Prepared BCNB1000 five-arm v1 campaign + baseline compatibility validation: ' + str(root))
    return plan


def all_records(plan):
    return {**plan['controls'], **plan['variants']}


def helper_plan(plan):
    return {**plan, 'variants': all_records(plan)}


def slurm_command(plan, variant, stage, minutes=None):
    if variant not in all_records(plan) or stage not in ('validation', 'full') or (variant == CONTROL and stage == 'full'):
        raise ValueError('Invalid v1 arm/stage')
    base = Path(all_records(plan)[variant]['root'])
    command = ['sbatch', '--parsable', '--account=' + plan['account'], '--qos=' + plan['qos'],
        '--partition=gpu', '--gres=gpu:rtx_pro_6000:1', '--constraint=rtxpro6000',
        '--cpus-per-task=8', '--mem=64G', '--time=' + str(minutes or 360),
        '--job-name=rules-v1-' + variant + '-' + stage,
        '--output=' + str(base / 'logs' / (stage + '-%A_%a.log'))]
    if stage == 'full':
        command.append('--array=0-9')
    return command + [str(base / (DATASET + '-' + stage + '.sh'))]


def submit_once(plan, variant, stage, queue, minutes=None):
    directory = Path(all_records(plan)[variant]['root'])
    receipt = directory / 'receipts' / (DATASET + '-' + stage + '.json')
    if receipt.exists():
        return read_json(receipt)['job_id']
    needed = 10 if stage == 'full' else 1
    if sum(row[3] == plan['qos'] for row in queue) + needed > plan['system_submit_limit']:
        return None
    command = slurm_command(plan, variant, stage, minutes)
    (directory / 'logs').mkdir(exist_ok=True)
    try:
        response = old.run_command(command)
    except subprocess.CalledProcessError as exc:
        if any(word in (exc.stderr or '').lower() for word in ('qosmaxsubmit', 'job violates accounting/qos', 'job submit limit')):
            return None
        raise
    job = response.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Invalid Slurm job id')
    atomic_json(receipt, {'job_id': job, 'command': command, 'submitted_at': old.now()})
    atomic_json(directory / 'receipts' / (DATASET + '-' + stage + '-verification.json'),
        {'job_id': job, 'verification': old.run_command(['scontrol', 'show', 'job', job, '-o'])})
    old.log('Accepted ' + variant + '/' + stage + ' job=' + job)
    return job


def verify_validation(plan, variant, job):
    # Reuse the existing strict runner/profile/scale/generation verification.
    adapted = helper_plan(plan)
    directory = Path(all_records(plan)[variant]['root']) / DATASET
    # Existing checker is smoke-folder-specific; use a local adapter without modifying old files.
    output = directory / 'validation'
    manifest = read_manifest(plan['datasets'][DATASET]['validation_manifest'])
    identity, run_hash = old.expected_run(adapted, variant, DATASET, manifest)
    if read_json(output / 'run_lock.json') != {**identity, 'run_hash': run_hash}:
        raise ValueError('Validation run-lock mismatch')
    profile = read_json(output / 'profile.json')
    required = {'passed': True, 'real_gpu': True, 'smoke': True, 'dataset': manifest['dataset'],
        'code_hash': identity['code_hash'], 'config_hash': digest(identity['config']),
        'manifest_hash': manifest['samples_hash'], 'run_hash': run_hash, 'slurm_job_id': str(job)}
    if any(profile.get(key) != value for key, value in required.items()):
        raise ValueError('Failed or stale GPU validation profile')
    fleet = read_json(Path(plan['root']) / 'gpu_fleet.json')
    if not all(word in profile['gpu'].lower() for word in ('rtx', 'pro', '6000')) or any(profile.get(k) != v for k, v in fleet.items()):
        raise ValueError('Validation GPU does not match RTX PRO6000 fleet')
    from data_processing.datasets import answer_label
    results = {}
    for row in manifest['samples']:
        result = read_json(output / 'results' / (digest(row['sample_id']) + '.json'))
        if (result.get('status') != 'ok' or result.get('sample_id') != row['sample_id']
                or result.get('run_hash') != run_hash or answer_label(result.get('pred_answer'), row['choices']) is None):
            raise ValueError('Invalid validation answer: ' + row['sample_id'])
        for event in result.get('process', []):
            decision = event.get('decision', {})
            if str(decision.get('zoom_recommendation', '')).lower() == 'yes':
                if type(decision.get('zoom_level')) is not int or decision['zoom_level'] not in (10, 20, 40):
                    raise ValueError('Illegal validation zoom scale')
        results[row['sample_id']] = result
    generation = read_json(output / 'resolved_generation.json')
    previous = read_json(next((Path(plan['baseline_root']) / DATASET / 'full').glob('shard_*/resolved_generation.json')))
    for key in ('executor', 'perceptor', 'overrides', 'image_processor', 'context_limit'):
        if generation[key] != previous[key]:
            raise ValueError('Generation changed: ' + key)
    baseline_results = read_json(Path(plan['datasets'][DATASET]['validation_manifest']).parent / 'baseline_validation_results.json')
    if variant == CONTROL:
        changed = [key for key in results if stable_trace(results[key]) != stable_trace(baseline_results[key])]
        if changed:
            raise ValueError('BASE_COMPAT model inputs, outputs or trajectory differ: ' + str(changed))
    if variant == 'C_ZOOM_LOOP':
        for key, baseline in baseline_results.items():
            zoom_events = [i for i, event in enumerate(baseline['process']) if 'zoom_finding' in event]
            if not zoom_events:
                if stable_trace(results[key]) != stable_trace(baseline):
                    raise ValueError('C-only no-Zoom validation changed: ' + key)
                continue
            first = zoom_events[0]
            process = results[key]['process']
            if len(process) <= first + 1 or process[first]['zoom_finding'] != baseline['process'][first]['zoom_finding']:
                raise ValueError('C first Zoom or follow-up review changed/missing: ' + key)
            child = process[first]['zoom_finding']['region']['region_id']
            if child not in {finding['region']['region_id'] for finding in process[first + 1]['findings']}:
                raise ValueError('C review omitted acquired child')
            original_calls = baseline['model_calls']
            calls = results[key]['model_calls']
            boundary = next((i for i, call in enumerate(original_calls)
                if call.get('region', {}).get('region_id') == child and call['stage'] == 'describe'), None)
            if boundary is None:
                raise ValueError('Baseline first Zoom model call missing')
            normalize = lambda rows: [{k: v for k, v in call.items() if k != 'seconds'} for call in rows]
            if normalize(calls[:boundary + 1]) != normalize(original_calls[:boundary + 1]):
                raise ValueError('C changed model-call prefix before first Zoom: ' + key)
            after = [call['stage'] for call in calls[boundary + 1:] if call['stage'] != 'summary']
            reflect_index = next((i for i, stage in enumerate(after) if stage == 'reflect'), None)
            if not after or after[0] != 'predict' or reflect_index is None or any(stage != 'predict' for stage in after[:reflect_index]):
                raise ValueError('C Zoom must be followed by Predict and Reflect: ' + key)
    return {'passed': True, 'profile': profile, 'metrics': score(manifest['samples'], results),
            'accuracy_is_gate': False, 'checked_at': old.now()}


def full_walltime(plan, variant):
    manifest = read_manifest(plan['datasets'][DATASET]['validation_manifest'])
    results = Path(plan['variants'][variant]['root']) / DATASET / 'validation/results'
    grouped = defaultdict(lambda: [0., 0.])
    for row in manifest['samples']:
        grouped[row['task']][0] += read_json(results / (digest(row['sample_id']) + '.json'))['agent_seconds']
        grouped[row['task']][1] += plan['datasets'][DATASET]['baseline_smoke_agent_seconds'][row['sample_id']]
    factor = max([1.] + [new / max(previous, 1.) for new, previous in grouped.values()])
    historical = max(plan['datasets'][DATASET]['historical_shard_seconds'])
    minimum = 720 if zoom_enabled(variant) else 360
    minutes = max(minimum, math.ceil(2 * factor * historical / 60 + 30))
    if minutes > plan['maximum_walltime_minutes']:
        raise ValueError('Full estimate exceeds partition limit; no truncation')
    return minutes, {'validation_task_agent_slowdown': factor, 'historical_max_shard_seconds': historical,
                     'method': 'max task QA ratio, 2x historical inference+preparation +30min'}


def controller(root, poll_seconds=30):
    plan = read_json(root / 'plan.json')
    with open(root / 'controller.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old.verify_frozen(plan)
        state_path = root / 'status.json'
        state = read_json(state_path) if state_path.exists() else {'created_at': old.now(), 'variants': {}}
        state.update(controller_pid=os.getpid(), controller_started_at=old.now())
        last_summary = None
        while True:
            queue = old.queue_rows()
            for variant in all_records(plan):
                status = state['variants'].setdefault(variant, {'stage': 'validation_pending', 'jobs': {}})
                if status['stage'] in ('complete', 'validation_passed', 'validation_failed', 'full_failed'):
                    continue
                try:
                    if status['stage'].startswith('validation'):
                        job = submit_once(plan, variant, 'validation', queue)
                        if not job:
                            continue
                        status['jobs']['validation'] = job
                        queue = old.queue_rows()
                        js = old.job_state(job, queue)
                        status.update(stage='validation_running', validation_job_state=js)
                        if js in old.TERMINAL_FAILURES:
                            raise ValueError('Slurm validation failed: ' + js)
                        if js == 'COMPLETED':
                            old.verify_frozen(plan)
                            gate = verify_validation(plan, variant, job)
                            atomic_json(Path(all_records(plan)[variant]['root']) / 'gate_decision.json', gate)
                            status.update(stage='validation_passed', validation_metrics=gate['metrics'])
                            old.log(variant + ': technical validation passed')
                    elif status['stage'].startswith('full'):
                        minutes, estimate = full_walltime(plan, variant)
                        atomic_json(Path(plan['variants'][variant]['root']) / DATASET / 'walltime_estimate.json', {'minutes': minutes, **estimate})
                        job = submit_once(plan, variant, 'full', queue, minutes)
                        if not job:
                            continue
                        status['jobs']['full'] = job
                        queue = old.queue_rows()
                        js = old.job_state(job, queue, 10)
                        status.update(stage='full_running', full_job_state=js)
                        if js in old.TERMINAL_FAILURES:
                            raise ValueError('Slurm full failed: ' + js)
                        if js == 'COMPLETED':
                            status['metrics'] = {DATASET: old.verify_full(helper_plan(plan), variant, DATASET)}
                            status['stage'] = 'complete'
                            old.log(variant + ': BCNB1000 complete')
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    status.update(stage='full_failed' if status['stage'].startswith('full') else 'validation_failed', error=str(exc))
                    old.log(variant + ': ' + status['stage'] + ': ' + str(exc))
            statuses = state['variants']
            if any(status['stage'] == 'validation_failed' for status in statuses.values()):
                state.update(complete=False, finished_at=old.now(), gate_error='Technical validation failed; no full arrays released')
                atomic_json(state_path, state)
                return state
            if (len(statuses) == 6 and all(status['stage'] == 'validation_passed' for status in statuses.values())):
                old.verify_frozen(plan)
                atomic_json(root / 'validation_gate.json', {'passed': True, 'checked_at': old.now(),
                    'arms': list(all_records(plan)), 'accuracy_is_gate': False})
                for variant in VARIANTS:
                    statuses[variant]['stage'] = 'full_pending'
                statuses[CONTROL]['stage'] = 'complete'
                old.log('All technical validations passed; releasing five arrays of ten shards without concurrency cap')
            state.update(updated_at=old.now(), system_queue_tasks=len(queue))
            atomic_json(state_path, state)
            summary = {variant: status['stage'] for variant, status in statuses.items()}
            if summary != last_summary:
                old.log('Campaign state: ' + json.dumps(summary))
                last_summary = summary
            if len(statuses) == 6 and all(status['stage'] in ('complete', 'full_failed') for status in statuses.values()):
                report = root / 'controller_source/slurm/report_rule_ablations_v1.py'
                subprocess.run([sys.executable, str(report), '--root', str(root)], check=True)
                state.update(complete=all(status['stage'] == 'complete' for status in statuses.values()), finished_at=old.now())
                atomic_json(state_path, state)
                return state
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'preview', 'run', 'status', 'report'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--baseline', default=str(old.BASELINE))
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.mode == 'prepare':
        prepare(root, Path(args.baseline).resolve())
    elif args.mode == 'run':
        if not 10 <= args.poll_seconds <= 60:
            parser.error('poll-seconds must be between 10 and 60')
        controller(root, args.poll_seconds)
    elif args.mode == 'status':
        print(json.dumps(read_json(root / 'status.json'), indent=2))
    elif args.mode == 'preview':
        plan = read_json(root / 'plan.json')
        old.verify_frozen(plan)
        for variant in all_records(plan):
            for stage in (('validation',) if variant == CONTROL else ('validation', 'full')):
                command = slurm_command(plan, variant, stage, 720 if stage == 'full' and zoom_enabled(variant) else 360)
                old.log(variant + '/' + stage + ': ' + old.run_command(command[:1] + ['--test-only'] + command[1:]).strip())
    else:
        subprocess.run([sys.executable, str(root / 'controller_source/slurm/report_rule_ablations_v1.py'), '--root', str(root)], check=True)


if __name__ == '__main__':
    main()
