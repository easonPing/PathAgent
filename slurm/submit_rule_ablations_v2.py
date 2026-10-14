"""Isolated six-arm BCNB rules-v2 campaign; no baseline GPU job is submitted."""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import difflib
import fcntl
import importlib.metadata
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import (ROOT, atomic_json, digest, load_config, read_json,
    read_manifest, sha256_file, write_manifest)
from eval.metrics import score
from slurm import submit_rule_ablations as old
from slurm.report_rule_ablations import atomic_text

DATASET = 'bcnb_1000_sample'
VARIANTS = ('R1_EVIDENCE', 'R2_DIFFERENTIAL', 'R3_ROI_VALIDITY',
            'R4_VISUAL_MISSING', 'R5_TARGET_REGION', 'ALL5')
PROTOCOL = 'pathology-rules-v2'
V1 = ROOT / 'runs/submissions/rules_v1_bcnb1000_20261008_052316'
FLEET = {'gpu': 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
         'gpu_memory_bytes': 101975851008}
SOURCE_FOLDERS = ('models', 'data_processing', 'data_preparation_script', 'scripts', 'eval', 'slurm')
REUSED_VARIANTS = ('R3_ROI_VALIDITY', 'R4_VISUAL_MISSING', 'R5_TARGET_REGION')
FRESH_VARIANTS = ('R1_EVIDENCE', 'R2_DIFFERENTIAL', 'ALL5')
FULL_COMPLETED = ('complete', 'completed_with_model_errors')
VALIDATION_READY = ('validation_passed', 'validation_completed_with_model_errors')


def source_files(root):
    return [root / 'pathagent.py'] + [p for folder in SOURCE_FOLDERS
                                    for p in sorted((root / folder).rglob('*.py'))]


def validation_samples(samples, seed=128):
    """Four distinct images per task, selected without predictions or labels."""
    tasks = sorted({row['task'] for row in samples})
    if len(tasks) != 7:
        raise ValueError('BCNB validation requires exactly seven tasks')
    chosen = set()
    for task in tasks:
        rows = sorted((row for row in samples if row['task'] == task),
                      key=lambda row: (digest([seed, task, row['sample_id']]), row['sample_id']))
        seen = set()
        for row in rows:
            if row['image_id'] in seen:
                continue
            seen.add(row['image_id'])
            chosen.add(row['sample_id'])
            if len(seen) == 4:
                break
        if len(seen) != 4:
            raise ValueError('Insufficient distinct validation images: ' + task)
    return [row for row in samples if row['sample_id'] in chosen]


def archived_baseline(baseline, full, verify_images=True):
    """Validate the historical archive, independently of changed workspace code."""
    previous = read_manifest(baseline / DATASET / 'full.json')
    if previous['samples'] != full['samples'] or previous['samples_hash'] != full['samples_hash']:
        raise ValueError('Baseline samples differ')
    results = old.load_results(baseline / DATASET / 'full')
    metrics = score(full['samples'], results)
    if not metrics['complete'] or metrics['invalid'] or metrics != read_json(baseline / DATASET / 'metrics.json'):
        raise ValueError('Baseline completeness or metrics differ')
    shards = sorted((baseline / DATASET / 'full').glob('shard_*'))
    if not shards:
        raise ValueError('Missing archived baseline shards')
    first_lock = read_json(shards[0] / 'run_lock.json')
    base = first_lock['config']
    inventory, result_hashes = {}, {}
    frozen_files = [baseline / 'source/pathagent.py']
    for folder in SOURCE_FOLDERS[:-1]:
        frozen_files.extend((baseline / 'source' / folder).rglob('*.py'))
    frozen_hash = digest({str(p.relative_to(baseline / 'source')): sha256_file(p)
                          for p in sorted(frozen_files)})
    if frozen_hash != first_lock['code_hash']:
        raise ValueError('Archived baseline source hash differs')
    for shard in shards:
        lock = read_json(shard / 'run_lock.json')
        identity = {k: v for k, v in lock.items() if k != 'run_hash'}
        if digest(identity) != lock['run_hash'] or lock['code_hash'] != frozen_hash:
            raise ValueError('Baseline run lock or source identity differs')
        for key in ('protocol', 'seed', 'models', 'algorithm', 'generation', 'preprocessing', 'runtime'):
            if lock['config'][key] != base[key]:
                raise ValueError('Baseline shard settings differ: ' + key)
        prompt = read_json(shard / 'prompt_manifest.json')
        if digest(prompt) != lock['prompt_manifest_hash']:
            raise ValueError('Baseline prompt hash differs')
        profile = read_json(shard / 'profile.json')
        if {k: profile[k] for k in FLEET} != FLEET:
            raise ValueError('Baseline GPU fleet differs')
        for image, record in read_json(shard / 'image_inventory.json').items():
            if image in inventory and inventory[image] != record:
                raise ValueError('Baseline image provenance differs')
            inventory[image] = record
        for path in (shard / 'results').glob('*.json'):
            result_hashes[read_json(path)['sample_id']] = sha256_file(path)
    expected = {row['image_id'] for row in full['samples']}
    if set(inventory) != expected:
        raise ValueError('Baseline image inventory incomplete')
    if verify_images:
        old.log('Verifying original SHA256 for ' + str(len(inventory)) + ' BCNB images')
        def verify(item):
            image, record = item
            if sha256_file(record['path']) != record['sha256']:
                raise ValueError('Original image changed: ' + image)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(verify, inventory.items()))
    from scripts.run_inference import check_assets
    check_assets(base)
    environment = read_json(shards[0] / 'environment.json')
    if Path(sys.executable).resolve() != Path(environment['executable']).resolve():
        raise ValueError('Use the original pathagent Python environment')
    for package, expected_version in environment['packages'].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected_version:
            raise ValueError('Environment differs: ' + package)
    return base, {'baseline': str(baseline), 'samples_hash': full['samples_hash'],
        'metrics': metrics, 'images': inventory, 'result_file_sha256': result_hashes,
        'source_hash': frozen_hash, 'packages': environment['packages'], 'gpu_fleet': FLEET,
        'generation_reference': str(shards[0] / 'resolved_generation.json'),
        'source_verification': 'archived source, not current workspace source'}, results


def verify_initial_manifest(path, base, full, history):
    manifest = read_json(path)
    if (manifest.get('kind') != 'frozen_initial_inputs'
            or manifest.get('samples_hash') != full['samples_hash']
            or set(manifest['images']) != set(history['images'])):
        raise ValueError('Frozen initial input identity/coverage differs')
    for key in ('seed', 'models', 'generation', 'preprocessing'):
        if manifest['contract'][key] != base[key]:
            raise ValueError('Frozen initial input contract differs: ' + key)
    if manifest['contract']['initial_magnification'] != base['algorithm']['initial_magnification']:
        raise ValueError('Frozen initial magnification differs')
    baseline_prompt = read_json(next((Path(history['baseline']) / DATASET / 'full').glob('shard_*/prompt_manifest.json')))
    for name, expected in manifest['contract'].get('perceptor_prompt_hashes', {}).items():
        if baseline_prompt['templates'][name]['sha256'] != expected:
            raise ValueError('Frozen Perceptor provenance differs: ' + name)
    files = {str(path): sha256_file(path)}
    for image, info in manifest['images'].items():
        if info['image_sha256'] != history['images'][image]['sha256']:
            raise ValueError('Frozen image identity differs: ' + image)
        directory = Path(info['directory'])
        for name in ('regions', 'observations'):
            file = directory / (name + '.json')
            if sha256_file(file) != info[name + '_sha256']:
                raise ValueError('Frozen initial input changed: ' + str(file))
            files[str(file)] = info[name + '_sha256']
        regions = read_json(directory / 'regions.json')['regions']
        observations = read_json(directory / 'observations.json')
        names = {region['region_id'] for region in regions}
        if names != set(observations['features']) or names != set(observations['descriptions']):
            raise ValueError('Frozen observations incomplete: ' + image)
    return {'manifest': str(path), 'sha256': files[str(path)], 'verified_files': files,
            'images': len(manifest['images']),
            'regions': sum(len(read_json(Path(info['directory']) / 'regions.json')['regions'])
                           for info in manifest['images'].values())}


def prepare(root, baseline=old.BASELINE, initial_manifest=V1 / 'initial_inputs.json', variants=VARIANTS,
            region_selection_policy='paper'):
    import yaml
    from models.region_selection import selection_manifest, selection_policy
    selection_policy({'region_selection_policy': region_selection_policy})
    root, baseline, initial_manifest = map(lambda p: Path(p).resolve(), (root, baseline, initial_manifest))
    if root.exists():
        raise ValueError('Use a new campaign directory')
    full = read_manifest(V1 / 'shared_manifests' / DATASET / 'full.json')
    current_full = read_manifest(ROOT / 'data/manifests' / DATASET / 'full.json')
    sampling = read_json(ROOT / 'data/raw/bcnb_1000_sample/sampling.json')
    if (len(full['samples']) != 1000 or not full.get('images_verified')
            or current_full['samples_hash'] != full['samples_hash']
            or [row['sample_id'] for row in full['samples']] != sampling['sample_ids']):
        raise ValueError('BCNB1000 manifest or sampling lock differs')
    base, history, baseline_results = archived_baseline(baseline, full)
    initial = verify_initial_manifest(initial_manifest, base, full, history)
    validation = validation_samples(full['samples'], base['seed'])
    original_shards = V1 / 'shared_manifests' / DATASET / 'shards'
    shards = [read_manifest(original_shards / ('shard_' + str(i) + '.json'))['samples'] for i in range(10)]
    old.validate_shards(full['samples'], shards, 10)
    root.mkdir(parents=True)
    shared = root / 'shared_manifests' / DATASET
    shared.mkdir(parents=True)
    shutil.copy2(V1 / 'shared_manifests' / DATASET / 'full.json', shared / 'full.json')
    shutil.copytree(original_shards, shared / 'shards')
    shutil.copy2(V1 / 'shared_manifests' / DATASET / 'sharding.json', shared / 'sharding.json')
    write_manifest(shared / 'validation.json', validation, dataset=full['dataset'], purpose='smoke',
        images_verified=True, availability_restricted=False, parent_samples_hash=full['samples_hash'],
        selection='seed128 SHA256(seed,task,sample_id), four distinct images per task; no outcome selection')
    atomic_json(shared / 'baseline_verification.json', history)
    atomic_json(shared / 'baseline_validation_results.json',
                {row['sample_id']: baseline_results[row['sample_id']] for row in validation})
    atomic_json(root / 'gpu_fleet.json', FLEET)
    plan = {'schema_version': 1, 'rule_version': 'rules-v2', 'root': str(root), 'created_at': old.now(),
        'seed': base['seed'], 'baseline_root': str(baseline), 'baseline_is_reused': True,
        'enabled_datasets': [DATASET], 'datasets': {DATASET: {
            'full_manifest': str(shared / 'full.json'), 'validation_manifest': str(shared / 'validation.json'),
            'smoke_manifest': str(shared / 'validation.json'), 'shard_manifests': str(shared / 'shards'),
            'samples_hash': full['samples_hash'], 'count': 1000, 'shards': 10,
            'baseline_expected_correct': 513, 'baseline_metrics': history['metrics']}},
        'initial_inputs': initial, 'variants': {}, 'account': old.ACCOUNT, 'qos': 'class',
        'gpu': 'RTX PRO 6000', 'array_throttle': None, 'system_submit_limit': 100,
        'maximum_walltime_minutes': 4320, 'validation_walltime_minutes': 480,
        'gate': 'All six technical validations and persisted manual image/output/trajectory review pass before any full arrays; accuracy is not a gate',
        'common_flow': {'post_zoom_reflect': True, 'zoom_topk': 5, 'max_iterations': 5,
                        'cumulative_evidence': True},
        'comparison_limitation': 'All six arms share top5 Zoom and post-Zoom review. No flow-only control; differences from baseline do not isolate each rule.',
        'files': dict(initial['verified_files'])}
    if region_selection_policy == 'max5':
        plan['region_selection_policy'] = 'max5'
        plan['common_flow'].update(zoom_topk=None, zoom_selection='all_valid_unvisited',
                                   low_region_selection='max(5, ceil(0.1N))',
                                   question_descriptions='all_selected')
        plan['comparison_limitation'] = (
            'All six arms share max5 low-region selection, all selected Question descriptions, '
            'all available Zoom children and post-Zoom review. No flow-only control; '
            'differences from baseline do not isolate each rule.')
    files = source_files(ROOT)
    frozen_baseline_source = baseline / 'source'
    code_diffs = []
    current_by_rel = {str(p.relative_to(ROOT)): p for p in files}
    baseline_by_rel = {str(p.relative_to(frozen_baseline_source)): p
                       for p in source_files(frozen_baseline_source) if p.exists()}
    for relative in sorted(set(current_by_rel) | set(baseline_by_rel)):
        previous = baseline_by_rel[relative].read_text() if relative in baseline_by_rel else ''
        current = current_by_rel[relative].read_text() if relative in current_by_rel else ''
        if previous != current:
            code_diffs.extend(difflib.unified_diff(previous.splitlines(True), current.splitlines(True),
                fromfile='baseline/' + relative, tofile='rules-v2/' + relative))
    (root / 'code_diff.patch').write_text(''.join(code_diffs), encoding='utf-8')
    baseline_templates = read_json(next((baseline / DATASET / 'full').glob('shard_*/prompt_manifest.json')))['templates']
    for variant in variants:
        directory = root / 'variants' / variant
        source = directory / 'source'
        for path in files:
            dest = source / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        config = old.absolute_config(base, DATASET)
        config['protocol'] = PROTOCOL
        config['rules_v2'] = {'variant': variant}
        config['algorithm'].update(zoom_return_to_loop=True, zoom_selected_children=5)
        if region_selection_policy == 'max5':
            config['algorithm'].update(region_selection_policy='max5', replenish_ratio=.10)
        config['runtime'].pop('initial_inputs_manifest', None)
        config['runtime'].pop('initial_inputs_hash', None)
        config['runtime'].update(rules_v2_initial_inputs_manifest=initial['manifest'],
            rules_v2_initial_inputs_hash=initial['sha256'], processed_root=str(root / 'shared_stage_cache'),
            rules_v2_evidence_image_root=str(root / 'shared_stage_cache/evidence_images'))
        config_path = directory / 'configs' / (DATASET + '.yaml')
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        load_config(config_path)
        info = old.snapshot_info(source, config_path)
        atomic_json(directory / 'prompt_manifest.json', info['prompts'])
        prompt_diffs = []
        for name, record in sorted(info['prompts']['templates'].items()):
            previous = baseline_templates.get(name, {}).get('text', '')
            current = record['text']
            if previous != current:
                prompt_diffs.extend(difflib.unified_diff(previous.splitlines(True), current.splitlines(True),
                    fromfile='baseline/' + name, tofile=variant + '/' + name))
        (directory / 'prompt_diff.patch').write_text(''.join(prompt_diffs), encoding='utf-8')
        out = directory / DATASET
        out.mkdir()
        shutil.copy2(shared / 'full.json', out / 'full.json')
        shutil.copy2(shared / 'validation.json', out / 'validation.json')
        shutil.copytree(shared / 'shards', out / 'manifests')
        for stage in ('validation', 'full'):
            manifest = shared / ('validation.json' if stage == 'validation' else 'shards')
            (directory / (DATASET + '-' + stage + '.sh')).write_text(old.job_script(
                source, config_path, manifest, out / stage, info['source_hash'], root / 'gpu_fleet.json',
                smoke=stage == 'validation', array=stage == 'full'), encoding='utf-8')
        plan['variants'][variant] = {'root': str(directory), 'source': str(source),
            'source_hash': info['source_hash'], 'prompt_manifest_hash': digest(info['prompts']),
            'rules': list(range(1, 6)) if variant == 'ALL5' else [int(variant[1])],
            'post_zoom_reflect': True, 'zoom_topk': None if region_selection_policy == 'max5' else 5,
            'datasets': {DATASET: {'config': str(config_path), 'config_hash': digest(config)}}}
        if region_selection_policy == 'max5':
            plan['variants'][variant]['region_selection'] = selection_manifest(config['algorithm'])
    for path in files:
        dest = root / 'controller_source' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for path in root.rglob('*'):
        if path.is_file():
            plan['files'][str(path)] = sha256_file(path)
    atomic_json(root / 'plan.json', plan)
    old.log('Prepared ' + str(len(variants)) + ' rules-v2 arms, no baseline GPU jobs: ' + str(root))
    return plan


@contextmanager
def campaign_locks(roots):
    """Hold predecessor locks throughout the successor controller's lifetime."""
    with ExitStack() as stack:
        for root in sorted({str(Path(value).resolve()) for value in roots}):
            handle = stack.enter_context(open(Path(root) / 'controller.lock', 'a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Controller remains active: ' + root) from exc
        yield


def verify_reuse_predecessor(plan, variants=REUSED_VARIANTS):
    if tuple(plan['variants']) != VARIANTS or plan.get('rule_version') != 'rules-v2':
        raise ValueError('Reuse requires the existing six-arm rules-v2 campaign')
    old.verify_frozen(plan)
    status_path = Path(plan['root']) / 'status.json'
    statuses = read_json(status_path).get('variants', {}) if status_path.exists() else {}
    jobs = {}
    for name, spec in plan['variants'].items():
        directory = Path(spec['root'])
        status = statuses.get(name) or {}
        if status.get('stage', '').startswith('full') or status.get('jobs', {}).get('full'):
            raise ValueError('Predecessor already entered full execution: ' + name)
        if (directory / 'receipts' / (DATASET + '-full.json')).exists() or (directory / DATASET / 'full').exists():
            raise ValueError('Predecessor already has full execution: ' + name)
        if name in variants:
            receipt = directory / 'receipts' / (DATASET + '-validation.json')
            value = read_json(receipt)
            if not str(value.get('job_id', '')).isdigit():
                raise ValueError('Missing real reused validation job: ' + name)
            jobs[name] = str(value['job_id'])
    return jobs


def _generic_implementation_identity(source):
    """Fingerprint all upstream files and every Perceptor-facing v2 method."""
    identity = {}
    paths = [p for folder in ('models', 'data_processing') for p in (Path(source) / folder).rglob('*.py')]
    for path in sorted(paths):
        relative = str(path.relative_to(source))
        if relative == 'models/rules_v2_prompts.py':
            # Effective templates and schemas are checked separately; provenance notes may change.
            continue
        parsed = ast.parse(path.read_text())
        if relative == 'models/rules_v2_inference.py':
            for node in parsed.body:
                if isinstance(node, ast.ClassDef) and node.name == 'RulesV2ModelBackend':
                    node.body = [item for item in node.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                 or item.name in ('__init__', 'set_case', 'describe', 'perceptor_user',
                                     'perceptor_system', 'save_evidence_image', 'describe_image')]
        identity[relative] = digest(ast.dump(parsed, include_attributes=False))
    return identity


def copy_verified_generic_cache(previous, current):
    """Copy only immutable Generic region records, never Question/Decision results."""
    old_spec, new_spec = previous['variants']['ALL5'], current['variants']['ALL5']
    if _generic_implementation_identity(Path(old_spec['source'])) != _generic_implementation_identity(Path(new_spec['source'])):
        raise ValueError('Upstream Perceptor implementation differs; Generic reuse is unsafe')
    old_config = load_config(old_spec['datasets'][DATASET]['config'])
    new_config = load_config(new_spec['datasets'][DATASET]['config'])
    for key in ('seed', 'models', 'generation', 'preprocessing', 'algorithm'):
        if old_config[key] != new_config[key]:
            raise ValueError('Generic reuse contract differs: ' + key)
    old_prompts = read_json(Path(old_spec['root']) / 'prompt_manifest.json')
    resolved = read_json(Path(old_spec['root']) / DATASET / 'validation/resolved_generation.json')['perceptor']
    expected_user = old_prompts['templates']['PERCEPTOR_USER']['text'].format(
        metadata='{metadata}', instruction=old_prompts['templates']['GENERIC_PROMPT']['text']) + '\nInput modality: H&E'
    for variant in FRESH_VARIANTS:
        prior_config = load_config(previous['variants'][variant]['datasets'][DATASET]['config'])
        fresh_config = load_config(current['variants'][variant]['datasets'][DATASET]['config'])
        for value in (prior_config, fresh_config):
            for name in ('processed_root', 'rules_v2_evidence_image_root'):
                value['runtime'].pop(name, None)
        if prior_config != fresh_config:
            raise ValueError('Decision-only fix changed scientific configuration: ' + variant)
        fresh = read_json(Path(current['variants'][variant]['root']) / 'prompt_manifest.json')
        prior = read_json(Path(previous['variants'][variant]['root']) / 'prompt_manifest.json')
        if fresh['templates'] != prior['templates']:
            raise ValueError('Effective prompts changed: ' + variant)
        for key in ('field_schemas', 'output_schemas', 'schemas'):
            if fresh.get(key) != prior.get(key):
                raise ValueError('Stage schemas changed: ' + variant)
    inputs = read_json(current['initial_inputs']['manifest'])
    old_cache = Path(old_config['runtime']['processed_root']) / 'generic'
    new_cache = Path(new_config['runtime']['processed_root']) / 'generic'
    verified, count = {}, 0
    for manifest_path in sorted(old_cache.glob('*/manifest.json')):
        meta = read_json(manifest_path)
        identity = meta.get('identity') or {}
        image = inputs['images'].get(str(meta.get('image_id')))
        if image is None or digest(identity) != manifest_path.parent.name or meta.get('identity_hash') != digest(identity):
            raise ValueError('Generic cache identity is invalid: ' + str(manifest_path))
        checks = {'image_sha256': image['image_sha256'], 'geometry_sha256': image['regions_sha256'],
            'seed': new_config['seed'], 'models': new_config['models']['perceptor'],
            'generation': new_config['generation'], 'generic_case_id': Path(image['source_cache']).name,
            'prompt_hashes': {name: old_prompts['templates'][name]['sha256']
                              for name in ('PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'PERCEPTOR_USER')}}
        if any(identity.get(key) != expected for key, expected in checks.items()):
            raise ValueError('Generic cache prompt/model/seed/geometry contract differs')
        if (identity.get('effective_generic_user_template') != expected_user
                or identity.get('resolved_perceptor_generation') != resolved):
            raise ValueError('Generic cache effective generation/user identity differs')
        regions = {digest(region): region for region in read_json(Path(image['directory']) / 'regions.json')['regions']}
        for path in sorted(manifest_path.parent.glob('*.json')):
            if path.name == 'manifest.json' or path.name.startswith('generated_calls_'):
                continue
            record = read_json(path)
            if record.get('identity_hash') != digest(identity) or regions.get(path.stem) != record.get('region'):
                raise ValueError('Generic cached region identity differs: ' + str(path))
            dest = new_cache / manifest_path.parent.name / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            verified[str(dest)] = sha256_file(dest)
            count += 1
    return {'kind': 'verified_generic_region_records_only', 'source_root': str(old_cache),
            'records': count, 'files': verified, 'implementation_identity': digest(_generic_implementation_identity(Path(new_spec['source'])))}


def prepare_reuse(root, reuse_root):
    """Three arms keep their old validation/full source; three receive a new freeze."""
    root, reuse_root = Path(root).resolve(), Path(reuse_root).resolve()
    if root == reuse_root or root.exists():
        raise ValueError('Use a new composed campaign directory')
    previous = read_json(reuse_root / 'plan.json')
    with campaign_locks([reuse_root]):
        jobs = verify_reuse_predecessor(previous)
        plan = prepare(root, previous['baseline_root'], previous['initial_inputs']['manifest'], FRESH_VARIANTS)
        for key in ('seed', 'initial_inputs', 'common_flow', 'baseline_root'):
            if plan[key] != previous[key]:
                raise ValueError('Reused scientific invariant differs: ' + key)
        for name in ('full_manifest', 'validation_manifest'):
            if read_manifest(plan['datasets'][DATASET][name]) != read_manifest(previous['datasets'][DATASET][name]):
                raise ValueError('Shared manifest differs: ' + name)
        for index in range(10):
            if sha256_file(Path(plan['datasets'][DATASET]['shard_manifests']) / ('shard_' + str(index) + '.json')) != sha256_file(Path(previous['datasets'][DATASET]['shard_manifests']) / ('shard_' + str(index) + '.json')):
                raise ValueError('Shared shard differs: ' + str(index))
        fresh = plan['variants']
        plan['variants'] = {name: deepcopy(previous['variants'][name] if name in REUSED_VARIANTS else fresh[name]) for name in VARIANTS}
        for name in REUSED_VARIANTS:
            (root / 'variants' / name).symlink_to(Path(plan['variants'][name]['root']), target_is_directory=True)
        # Keep hashes of actual old immutable files. Mutable receipts/results/status are never frozen or rewritten.
        reused_roots = [Path(previous['variants'][name]['root']) for name in REUSED_VARIANTS]
        for path, expected in previous['files'].items():
            actual = Path(path)
            if any(actual.is_relative_to(directory) for directory in reused_roots) or not actual.is_relative_to(reuse_root / 'variants'):
                plan['files'][path] = expected
        plan['files'][str(reuse_root / 'plan.json')] = sha256_file(reuse_root / 'plan.json')
        generic = copy_verified_generic_cache(previous, plan)
        plan['files'].update(generic['files'])
        plan['reuse'] = {'root': str(reuse_root), 'plan_sha256': sha256_file(reuse_root / 'plan.json'),
                        'variants': list(REUSED_VARIANTS), 'validation_jobs': jobs,
                        'controller_lock_roots': [str(reuse_root)], 'generic_cache': generic}
        for name, spec in plan['variants'].items():
            spec['source_version'] = {'campaign_root': str(reuse_root if name in REUSED_VARIANTS else root),
                'reused_validation': name in REUSED_VARIANTS, 'validation_root': str(Path(spec['root']) / DATASET / 'validation'),
                'full_root': str(Path(spec['root']) / DATASET / 'full'), 'validation_source_hash': spec['source_hash'],
                'full_source_hash': spec['source_hash'], 'config_hash': spec['datasets'][DATASET]['config_hash']}
        plan['source_version_note'] = 'R3/R4/R5 use their original frozen validation AND full source/config. R1/R2/ALL5 use the new decision-source fixes; historical validation never certifies new code.'
        (root / 'source_versions.json').write_text(json.dumps({name: spec['source_version'] for name, spec in plan['variants'].items()}, indent=2) + '\n')
        plan['files'][str(root / 'source_versions.json')] = sha256_file(root / 'source_versions.json')
        atomic_json(root / 'plan.json', plan)
        statuses = {name: {'stage': 'validation_pending', 'jobs': {'validation': jobs[name]} if name in jobs else {},
                           'reused_validation': name in jobs} for name in VARIANTS}
        state = {'created_at': old.now(), 'variants': statuses, 'complete': False}
        atomic_json(root / 'status.json', state)
        old.verify_frozen(plan)
        write_pending_html(plan, state)
        return plan


def prepare_controller_update(root, reuse_root):
    """Keep all six real model runs; update only their controller and report."""
    root, reuse_root = Path(root).resolve(), Path(reuse_root).resolve()
    if root == reuse_root or root.exists():
        raise ValueError('Use a new controller campaign directory')
    previous = read_json(reuse_root / 'plan.json')
    lock_roots = sorted({str(reuse_root), *previous.get('reuse', {}).get('controller_lock_roots', [])})
    with campaign_locks(lock_roots):
        jobs = verify_reuse_predecessor(previous, VARIANTS)
        plan = deepcopy(previous)
        plan.update(root=str(root), files=deepcopy(previous['files']))
        root.mkdir()
        (root / 'variants').mkdir()
        for name, spec in plan['variants'].items():
            (root / 'variants' / name).symlink_to(Path(spec['root']), target_is_directory=True)
        # Dependencies remain byte-identical to the previous controller snapshot.
        shutil.copytree(reuse_root / 'controller_source', root / 'controller_source',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        diffs = []
        for name in ('submit_rule_ablations_v2.py', 'report_rule_ablations_v2.py'):
            prior = reuse_root / 'controller_source/slurm' / name
            current = ROOT / 'slurm' / name
            diffs.extend(difflib.unified_diff(prior.read_text().splitlines(True),
                current.read_text().splitlines(True), fromfile='previous-controller/' + name,
                tofile='new-controller/' + name))
            shutil.copy2(current, root / 'controller_source/slurm' / name)
        (root / 'controller_update_diff.patch').write_text(''.join(diffs))
        for name in ('code_diff.patch', 'source_versions.json', 'shared_manifests'):
            target = reuse_root / name
            if target.exists():
                (root / name).symlink_to(target, target_is_directory=target.is_dir())
        verification = root / 'verification'
        verification.mkdir()
        for path in (reuse_root / 'verification').glob('manual*.json'):
            if path.name != 'manual_validation_review.json':
                (verification / path.name).symlink_to(path.resolve())
        plan['reuse'] = {'kind': 'controller_report_only', 'root': str(reuse_root),
            'plan_sha256': sha256_file(reuse_root / 'plan.json'), 'variants': list(VARIANTS),
            'validation_jobs': jobs, 'controller_lock_roots': lock_roots}
        plan['controller_update'] = {'kind': 'controller_report_only',
            'previous_root': str(reuse_root), 'model_source_config_roots_unchanged': True,
            'manual_validation_note': 'Per-arm actual human reviews are linked; the aggregate release record must reference this new plan hash.'}
        plan['validation_readiness_policy'] = {
            'criterion': 'Exhaustive fixed 28-case validation and actual available-artifact human review; accuracy is not a gate.',
            'model_format_errors': 'Known bounded format errors remain failures in the 28-case denominator and are listed explicitly; they do not cause clinical prompt retuning.',
            'blocking_errors': 'Source/config/GPU/generation/context, acquisition geometry, action and loop budget violations block release.',
            'scheduler_guard_exception': 'Only validation FAILED|1:0 with the exact frozen runner post-result smoke-guard log and exhaustive 28 audited known format errors can be ready; actual FAILED state/exit/log proof remains recorded. Full FAILED, cancellation, timeout and OOM never qualify.',
            'partial_trajectory': 'The frozen runner omits partial process on format failure; saved raw calls and real image metadata are audited, without claiming a successful trajectory or legal Final.',
            'manual_acknowledgement_required': True}
        plan['files'][str(reuse_root / 'plan.json')] = sha256_file(reuse_root / 'plan.json')
        for path in (root / 'controller_source').rglob('*.py'):
            plan['files'][str(path)] = sha256_file(path)
        plan['files'][str(root / 'controller_update_diff.patch')] = sha256_file(root / 'controller_update_diff.patch')
        atomic_json(root / 'plan.json', plan)
        state = {'created_at': old.now(), 'complete': False, 'variants': {
            name: {'stage': 'validation_pending', 'jobs': {'validation': jobs[name]},
                   'reused_validation': True} for name in VARIANTS}}
        atomic_json(root / 'status.json', state)
        old.verify_frozen(plan)
        write_pending_html(plan, state)
        return plan


def slurm_command(plan, variant, stage):
    if variant not in VARIANTS or stage not in ('validation', 'full'):
        raise ValueError('Invalid rules-v2 arm/stage')
    directory = Path(plan['variants'][variant]['root'])
    minutes = 480 if stage == 'validation' else 4320
    command = ['sbatch', '--parsable', '--account=' + plan['account'], '--qos=' + plan['qos'],
        '--partition=gpu', '--gres=gpu:rtx_pro_6000:1', '--constraint=rtxpro6000',
        '--cpus-per-task=8', '--mem=64G', '--time=' + str(minutes),
        '--job-name=rules-v2-' + variant + '-' + stage,
        '--output=' + str(directory / 'logs' / (stage + '-%A_%a.log'))]
    if stage == 'full':
        command.append('--array=0-9')
    return command + [str(directory / (DATASET + '-' + stage + '.sh'))]


def submit_once(plan, variant, stage, queue):
    directory = Path(plan['variants'][variant]['root'])
    receipt = directory / 'receipts' / (DATASET + '-' + stage + '.json')
    if receipt.exists():
        return read_json(receipt)['job_id']
    needed = 10 if stage == 'full' else 1
    if sum(len(row) > 3 and row[3] == plan['qos'] for row in queue) + needed > plan['system_submit_limit']:
        return None
    command = slurm_command(plan, variant, stage)
    (directory / 'logs').mkdir(exist_ok=True)
    try:
        response = old.run_command(command)
    except subprocess.CalledProcessError as exc:
        if any(word in (exc.stderr or '').lower() for word in
               ('qosmaxsubmit', 'job violates accounting/qos', 'job submit limit')):
            return None
        raise
    job = response.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Invalid Slurm job ID')
    atomic_json(receipt, {'job_id': job, 'command': command, 'submitted_at': old.now()})
    atomic_json(directory / 'receipts' / (DATASET + '-' + stage + '-verification.json'),
                {'job_id': job, 'verification': old.run_command(['scontrol', 'show', 'job', job, '-o'])})
    old.log('Accepted ' + variant + '/' + stage + ' job=' + job)
    return job


def verify_trajectory(result, variant):
    """Hard execution invariants; pathology accuracy is deliberately not a gate."""
    events = result.get('process') or []
    if not events and result.get('stop_reason') == 'no_valid_low_regions' and variant in ('R3_ROI_VALIDITY', 'ALL5'):
        summary = result.get('quality_summary') or {}
        total = result.get('initial_region_count')
        if (not total or result.get('total_evidence_regions') != 0 or result.get('total_examined_regions') != 0
                or summary.get('checked') != total or summary.get('hard_rejected') != total):
            raise ValueError('Empty valid-ROI trajectory lacks recorded filtering evidence')
        return
    if not events or len(events) > 5:
        raise ValueError('Missing trajectory or iteration budget exceeded')
    acquisitions = 0
    for index, event in enumerate(events):
        action = event.get('actual_action', event.get('action'))
        executed = event.get('action_executed', action in ('zoom', 'explore'))
        if executed and action in ('zoom', 'explore'):
            acquisitions += 1
            if index == len(events) - 1 or event.get('iteration', index + 1) >= 5:
                raise ValueError('Acquisition lacks a following review')
        decision = event.get('decision') or {}
        if str(decision.get('zoom_recommendation', '')).lower() == 'yes':
            scale = decision.get('zoom_level', decision.get('recommended_zoom_level'))
            if type(scale) not in (int, float) or scale not in (10, 20, 40):
                raise ValueError('Illegal requested magnification')
        children = event.get('zoom_findings') or []
        if action == 'zoom' and executed:
            if not 1 <= len(children) <= 5:
                raise ValueError('Zoom must save between one and five children')
            ids = [finding['region']['region_id'] for finding in children]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate children within Zoom batch')
            next_ids = {finding['region']['region_id'] for finding in events[index + 1].get('findings', [])}
            if not set(ids).issubset(next_ids):
                raise ValueError('Post-Zoom review omitted acquired children')
            next_decision = events[index + 1].get('decision') or {}
            if 'answer' not in next_decision or 'sufficient' not in next_decision:
                raise ValueError('Post-Zoom Predict/Reflect output missing')
            target = event.get('requested_target_region')
            if target and any(finding['region'].get('parent_id') != target for finding in children):
                raise ValueError('Requested target differs from executed parent')
            if any(finding['region'].get('scale') not in (10, 20, 40) for finding in children):
                raise ValueError('Illegal executed magnification')
        if variant in ('R1_EVIDENCE', 'ALL5'):
            if decision.get('scale_failures') and str(decision.get('sufficient')).lower() != 'no':
                raise ValueError('Scale failure accepted by Reflect')
        if variant in ('R3_ROI_VALIDITY', 'ALL5'):
            quality = result.get('roi_quality') or {}
            if any(quality.get(finding.get('region', {}).get('region_id'), {}).get('hard_reject')
                   for finding in (event.get('findings') or []) + children):
                raise ValueError('Hard-rejected ROI admitted as evidence')
    if acquisitions > 4:
        raise ValueError('Acquisition budget exceeded')
    if result.get('audit_failures'):
        raise ValueError('Runtime audit failures: ' + str(result['audit_failures']))


def verify_validation(plan, variant, job):
    output = Path(plan['variants'][variant]['root']) / DATASET / 'validation'
    manifest = read_manifest(plan['datasets'][DATASET]['validation_manifest'])
    identity, run_hash = old.expected_run(plan, variant, DATASET, manifest)
    if read_json(output / 'run_lock.json') != {**identity, 'run_hash': run_hash}:
        raise ValueError('Validation run lock differs')
    profile = read_json(output / 'profile.json')
    required = {'real_gpu': True, 'smoke': True, 'dataset': manifest['dataset'],
        'code_hash': identity['code_hash'], 'config_hash': digest(identity['config']),
        'manifest_hash': manifest['samples_hash'], 'run_hash': run_hash, 'slurm_job_id': str(job), **FLEET}
    if any(profile.get(key) != value for key, value in required.items()):
        raise ValueError('Failed/stale GPU validation profile')
    from data_processing.datasets import answer_label
    results = {}
    for row in manifest['samples']:
        result = read_json(output / 'results' / (digest(row['sample_id']) + '.json'))
        if (result.get('sample_id') != row['sample_id'] or result.get('run_hash') != run_hash
                or (result.get('status') == 'ok' and answer_label(result.get('pred_answer'), row['choices']) is None)):
            raise ValueError('Invalid validation result: ' + row['sample_id'])
        results[row['sample_id']] = result
    contract = audit_full_results(plan, variant, results)
    metrics = score(manifest['samples'], results)
    if profile.get('passed') is not (metrics['complete'] and metrics['invalid'] == 0):
        raise ValueError('GPU profile validity disagrees with actual validation results')
    generation = read_json(output / 'resolved_generation.json')
    previous = read_json(next((Path(plan['baseline_root']) / DATASET / 'full').glob('shard_*/resolved_generation.json')))
    for key in ('executor', 'perceptor', 'overrides', 'image_processor', 'context_limit'):
        if generation[key] != previous[key]:
            raise ValueError('Generation differs from baseline: ' + key)
    all_valid = metrics['invalid'] == 0
    metrics['execution_contract'] = contract
    return {'ready': True, 'passed': all_valid, 'all_answers_valid': all_valid,
            'completion_status': 'validation_passed' if all_valid else 'validation_completed_with_model_errors',
            'checked_at': old.now(), 'accuracy_is_gate': False,
            'profile': profile, 'metrics': metrics, 'execution_contract': contract,
            'trajectory_checks': {'successful_questions': contract['verified_successful_trajectories'],
                'invalid_without_serialized_partial_trajectory': contract['invalid_without_serialized_partial_trajectory'],
                'failures': []}}


def verify_failed_validation_smoke_exit(plan, variant, job):
    """Recognize only the frozen runner's post-result smoke guard, never an infrastructure failure."""
    try:
        accounting = old.run_command(['sacct', '-n', '-P', '-j', str(job), '--format=JobID%64,State,ExitCode'])
    except subprocess.CalledProcessError as exc:
        raise ValueError('Unable to verify failed validation Slurm exit') from exc
    records = []
    for line in accounting.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 3:
            records.append({'job_id': fields[0], 'state': fields[1].split()[0].rstrip('+'), 'exit_code': fields[2]})
    allocations = [record for record in records if record['job_id'] == str(job)]
    if len(allocations) != 1 or allocations[0]['state'] != 'FAILED' or allocations[0]['exit_code'] != '1:0':
        raise ValueError('Failed validation is not the exact FAILED|1:0 smoke exit')
    steps = [record for record in records if record['job_id'].startswith(str(job) + '.')]
    if any((record['state'], record['exit_code']) not in (('FAILED', '1:0'), ('COMPLETED', '0:0')) for record in steps):
        raise ValueError('Failed validation has an infrastructure/other Slurm step failure')
    spec = plan['variants'][variant]
    receipt = read_json(Path(spec['root']) / 'receipts' / (DATASET + '-validation.json'))
    if str(receipt['job_id']) != str(job):
        raise ValueError('Failed validation receipt job differs')
    outputs = [value.split('=', 1)[1] for value in receipt['command'] if value.startswith('--output=')]
    if len(outputs) != 1:
        raise ValueError('Cannot locate the actual failed validation log')
    log_path = Path(outputs[0].replace('%A', str(job)).replace('%j', str(job)).replace('%a', '4294967294'))
    runner = Path(spec['source']) / 'scripts/run_inference.py'
    guard = "raise RuntimeError('Real GPU smoke failed: missing or invalid results')"
    guard_lines = [index for index, line in enumerate(runner.read_text().splitlines(), 1) if line.strip() == guard]
    log = log_path.read_text()
    marker = 'RuntimeError: Real GPU smoke failed: missing or invalid results'
    frame = f'File "{runner}", line {guard_lines[0]}, in main' if len(guard_lines) == 1 else None
    if not frame or not log.rstrip().endswith(marker) or frame not in log or guard not in log:
        raise ValueError('Failed validation log lacks the exact frozen runner smoke-guard termination')
    old.verify_frozen(plan)
    gate = verify_validation(plan, variant, job)
    metrics = gate['metrics']
    if (gate.get('ready') is not True or gate.get('all_answers_valid') is not False
            or metrics['expected'] != 28 or metrics['received'] != 28 or metrics['missing'] != 0
            or metrics['invalid'] < 1 or gate['profile'].get('passed') is not False
            or gate['execution_contract']['model_protocol_invalid_count'] != metrics['invalid']):
        raise ValueError('Failed validation smoke exit lacks exhaustive audited known model-format failures')
    gate['scheduler_failure_acceptance'] = {'state': 'FAILED', 'exit_code': '1:0',
        'reason': 'Frozen runner smoke guard rejected known model-format failures after saving all 28 results; finite contract readiness independently verified.',
        'accounting_command': ['sacct', '-n', '-P', '-j', str(job), '--format=JobID%64,State,ExitCode'],
        'accounting_raw': accounting, 'log_path': str(log_path), 'log_sha256': sha256_file(log_path),
        'frozen_runner_path': str(runner), 'frozen_runner_sha256': sha256_file(runner), 'termination_marker': marker}
    return gate


def model_format_invalid(result):
    """Only observed output-format failures are scoreable model errors."""
    error = result.get('error', '')
    match = re.fullmatch(r'Invalid (predict|reflect|missing_info|final) response after 2 attempts', error)
    calls = result.get('model_calls') or []
    if match:
        stage = match.group(1)
        return (result.get('error_stage') == stage and len(calls) >= 2
                and all(call.get('stage') == stage and isinstance(call.get('raw_response'), str)
                        for call in calls[-2:]))
    if error == 'Invalid tissue-status Question description after two attempts':
        return (result.get('error_stage') == 'describe' and len(calls) >= 2
                and all(call.get('stage') == 'describe' and isinstance(call.get('raw_response'), str)
                        for call in calls[-2:]))
    return (error == 'Perceptor returned an empty description'
            and result.get('error_stage') == 'describe' and bool(calls)
            and calls[-1].get('stage') == 'describe'
            and isinstance(calls[-1].get('raw_response'), str))


def initial_region_registry(plan, results):
    if 'initial_inputs' not in plan:
        return {}  # Legacy synthetic report fixtures have no acquisition metadata.
    inputs = read_json(plan['initial_inputs']['manifest'])
    registry = {}
    for image_id in {str(result['image_id']) for result in results.values()}:
        info = inputs['images'][image_id]
        path = Path(info['directory']) / 'regions.json'
        if sha256_file(path) != info['regions_sha256']:
            raise ValueError('Frozen Region geometry changed: ' + image_id)
        registry[image_id] = {region['region_id']: region for region in read_json(path)['regions']}
    return registry


def verify_result_geometry(result, initial=None):
    """Verify actual acquisition metadata, including invalid cases' saved calls."""
    from data_processing.regions import Region, zoom_regions
    observed = {}
    regions = [call['region'] for call in result.get('model_calls', []) if call.get('region')]
    for event in result.get('process') or []:
        regions.extend(finding['region'] for finding in
            (event.get('findings') or []) + (event.get('zoom_findings') or []) if finding.get('region'))
    for region in regions:
        name = region['region_id']
        if name in observed and observed[name] != region:
            raise ValueError('Conflicting actual Region geometry: ' + name)
        observed[name] = region
    for call in result.get('model_calls') or []:
        if call.get('region') and call.get('region_metadata') and Region(**call['region']).metadata() != call['region_metadata']:
            raise ValueError('Serialized actual Region differs from raw acquisition metadata')
    parents = {**(initial or {}), **{key: value for key, value in observed.items() if not value.get('parent_id')}}
    for call in result.get('model_calls') or []:
        # Failed Perceptor attempts retain program-supplied metadata, not a normalized Region dict.
        metadata = call.get('region_metadata')
        if call.get('stage') != 'describe' or call.get('region') or not metadata:
            continue
        name = metadata.split(';', 1)[0].removeprefix('Region=')
        candidate = parents.get(name)
        if candidate is None and '/z' in name:
            parent_id, tail = name.rsplit('/z', 1)
            parent = parents.get(parent_id)
            if parent:
                try:
                    target = float(tail.split('/', 1)[0])
                    candidate = next((r.to_dict() for r in zoom_regions(Region(**parent), target)
                                      if r.region_id == name), None)
                except (ValueError, KeyError):
                    candidate = None
        if candidate is None or Region(**candidate).metadata() != metadata:
            raise ValueError('Failed Perceptor acquisition metadata differs from frozen geometry: ' + name)
        if name in observed and observed[name] != candidate:
            raise ValueError('Conflicting actual Region geometry: ' + name)
        observed[name] = candidate
    for name, region in observed.items():
        parent_id = region.get('parent_id')
        if parent_id is None:
            if region.get('scale') != 5:
                raise ValueError('Illegal acquired parent magnification: ' + name)
            if initial is not None and initial.get(name) != region:
                raise ValueError('Actual parent differs from frozen Region: ' + name)
            continue
        if region.get('scale') not in (10, 20, 40):
            raise ValueError('Illegal acquired child magnification: ' + name)
        parent = parents.get(parent_id)
        if parent is None:
            raise ValueError('Actual Zoom parent geometry unavailable: ' + name)
        candidates = {child.region_id: child.to_dict() for child in
                      zoom_regions(Region(**parent), float(region['scale']))}
        if candidates.get(name) != region:
            raise ValueError('Executed Zoom coordinates differ from original parent: ' + name)
    return observed


def audit_raw_call_contracts(result, observed, context_limit=None):
    """Check provable partial execution budgets without fabricating a process."""
    blocks = Counter()
    stage_calls = Counter()
    previous = None
    consecutive = 0
    child_batches, children, low = [], set(), set()
    batch = set()
    checked_context = 0
    for call in result.get('model_calls') or []:
        stage = call.get('stage')
        stage_calls[stage] += 1
        consecutive = consecutive + 1 if stage == previous else 1
        if stage in ('predict', 'reflect', 'missing_info', 'final') and consecutive > 2:
            raise ValueError('Raw calls exceed the single format retry contract')
        if stage != previous:
            if batch:
                child_batches.append(batch)
                batch = set()
            if stage in ('predict', 'reflect', 'missing_info'):
                blocks[stage] += 1
        if context_limit is not None and call.get('input_tokens') is not None:
            tokens = call['input_tokens']
            cap = (call.get('generation') or {}).get('max_new_tokens')
            if type(tokens) is not int or type(cap) is not int or tokens < 0 or cap <= 0 or tokens + cap > context_limit:
                raise ValueError('Raw model call exceeds frozen context contract')
            checked_context += 1
        if stage == 'describe':
            region = call.get('region')
            name = region.get('region_id') if region else str(call.get('region_metadata', '')).split(';', 1)[0].removeprefix('Region=')
            region = observed.get(name)
            if region and region.get('parent_id'):
                children.add(name)
                batch.add(name)
            elif region:
                low.add(name)
        previous = stage
    if batch:
        child_batches.append(batch)
    if stage_calls['final'] > 2:
        raise ValueError('Raw calls exceed the single Final format retry contract')
    if any(blocks[stage] > 5 or stage_calls[stage] > 10 for stage in blocks) or len(children) > 20 or len(low) > 25 or len(child_batches) > 4 or any(len(item) > 5 for item in child_batches):
        raise ValueError('Raw calls prove iteration/acquisition budget violation')
    return checked_context


def audit_full_results(plan, variant, results):
    """Pure contract audit; invalid model formatting never drops a sample."""
    registries = initial_region_registry(plan, results)
    manifest_path = plan.get('datasets', {}).get(DATASET, {}).get('full_manifest')
    samples = {row['sample_id']: row for row in read_manifest(manifest_path)['samples']} if manifest_path else {}
    from data_processing.datasets import answer_label
    spec = plan.get('variants', {}).get(variant, {})
    generation_path = Path(spec.get('root', '.')) / DATASET / 'validation/resolved_generation.json'
    context_limit = read_json(generation_path)['context_limit'] if generation_path.exists() else None
    invalid, valid, partial, context_calls = [], 0, [], 0
    for sid, result in results.items():
        if result.get('audit_failures'):
            raise ValueError('Runtime audit failures: ' + str(result['audit_failures']))
        observed = verify_result_geometry(result, registries.get(str(result.get('image_id'))))
        context_calls += audit_raw_call_contracts(result, observed, context_limit)
        if result.get('status') == 'ok':
            if sid in samples and samples[sid].get('choices') and answer_label(
                    result.get('pred_answer'), samples[sid]['choices']) is None:
                raise ValueError('Successful result contains illegal Final option: ' + sid)
            verify_trajectory(result, variant)
            valid += 1
        elif result.get('status') == 'invalid' and model_format_invalid(result):
            # The frozen runner retains calls but does not serialize its partial local trajectory.
            if result.get('process'):
                verify_trajectory(result, variant)
            else:
                partial.append(sid)
            invalid.append(sid)
        else:
            raise ValueError('Non-format execution/protocol failure: ' + sid + ': ' + str(result.get('error')))
    return {'passed': True, 'verified_successful_trajectories': valid,
        'model_protocol_invalid_count': len(invalid), 'model_protocol_invalid_sample_ids': sorted(invalid),
        'invalid_without_serialized_partial_trajectory': sorted(partial),
        'geometry_checked': True,
        'raw_context_calls_checked': context_calls, 'context_limit': context_limit,
        'completion_status': 'completed_with_model_errors' if invalid else 'complete'}


def verify_full(plan, variant):
    metrics = old.verify_full(plan, variant, DATASET)
    results = old.load_results(Path(plan['variants'][variant]['root']) / DATASET / 'full')
    contract = audit_full_results(plan, variant, results)
    metrics.update(execution_contract=contract, completion_status=contract['completion_status'])
    atomic_json(Path(plan['variants'][variant]['root']) / DATASET / 'metrics.json', metrics)
    return metrics


def release_gate(plan, statuses):
    return (set(statuses) == set(VARIANTS)
            and all(statuses[variant]['stage'] in VALIDATION_READY for variant in VARIANTS))


def manual_validation_gate(plan, statuses=None):
    path = Path(plan['root']) / 'verification/manual_validation_review.json'
    if not path.exists():
        return None
    review = read_json(path)
    if (review.get('complete') is not True or review.get('plan_hash') != digest(plan)
            or set(review.get('reviewed_variants') or []) != set(VARIANTS)):
        return None
    scope = review.get('scope') or {}
    if any(scope.get(name) is not True for name in ('images', 'model_outputs', 'trajectories')):
        raise ValueError('Manual validation scope must cover images, model outputs and trajectories')
    paths = {str(Path(value).resolve()) for value in review.get('reviewed_paths') or []}
    expected = {str((Path(plan['variants'][variant]['root']) / DATASET / 'validation').resolve())
                for variant in VARIANTS}
    if not expected.issubset(paths) or not all(Path(value).is_dir() for value in expected):
        raise ValueError('Manual validation reviewed_paths must contain all six validation directories')
    if review.get('execution_contract_passed') is False:
        raise ValueError('Manual review explicitly rejects the execution contract')
    if plan.get('validation_readiness_policy', {}).get('manual_acknowledgement_required'):
        if review.get('execution_contract_passed') is not True or review.get('available_artifacts_reviewed') is not True:
            raise ValueError('Manual review must explicitly accept execution contracts and complete available-artifact review')
    state_path = Path(plan['root']) / 'status.json'
    if statuses is None:
        statuses = read_json(state_path).get('variants', {}) if state_path.exists() else {}
    invalid = {(name, sid) for name, status in statuses.items() for sid in
               status.get('validation_metrics', {}).get('execution_contract', {}).get('model_protocol_invalid_sample_ids', [])}
    if invalid:
        acknowledgement = review.get('model_format_failure_acknowledgement') or {}
        required = ('accepted_for_fixed_rule_evaluation', 'available_artifacts_reviewed', 'missing_partial_trajectories_acknowledged')
        cases = {(item.get('variant'), item.get('sample_id')) for item in acknowledgement.get('invalid_cases', [])}
        if any(acknowledgement.get(key) is not True for key in required) or cases != invalid:
            raise ValueError('Known model-format failures require explicit available-artifact and missing-trajectory acknowledgement')
    return {'passed': True, 'path': str(path), 'sha256': sha256_file(path), 'review': review}


def generate_report(root):
    report = root / 'controller_source/slurm/report_rule_ablations_v2.py'
    subprocess.run([sys.executable, str(report), '--root', str(root)], check=True)


def write_pending_html(plan, state):
    """A progress view only: incomplete outputs never become accuracy estimates."""
    root = Path(plan['root'])
    escape = lambda value: html.escape(str(value), quote=True)
    statuses = state.get('variants') or {}
    technical_path, gate_path = root / 'technical_validation_gate.json', root / 'validation_gate.json'
    technical_record = read_json(technical_path) if technical_path.exists() else {}
    technical_passed = technical_record.get('ready', technical_record.get('passed')) is True
    manual_passed = gate_path.exists() and read_json(gate_path).get('manual_review', {}).get('passed') is True
    technical = ('六组有限验证已完成（含模型格式失败）' if technical_record.get('all_answers_valid') is False
                 else '六组技术验证通过') if technical_passed else '等待六组技术验证'
    manual = '可用图片、输出及轨迹已人工审查' if manual_passed else '等待可用图片、输出及轨迹人工审查'
    body = '<h1>BCNB1000 · rules-v2 运行进度</h1><p>' + escape(state.get('updated_at', old.now())) + '</p>'
    body += '<p>' + technical + ' · ' + manual + '</p><p class="note">原 baseline 复用。六组验证及人工审查通过后启动正式运行；未完成结果仅显示进度。</p>'
    if plan.get('validation_readiness_policy'):
        body += '<p class="note">固定28题验证要求结果齐全、真实GPU和冻结执行合同合规，正确率不作gate。已知有限模型格式失败保留在28题分母、单列记录并人工确认可用原文和图片；未序列化的部分轨迹不宣称审查成功，也不补造Final。基础设施、context、坐标、预算或源码合同违规阻止正式运行。</p>'
    invalid_cases = technical_record.get('model_format_invalid_cases') or []
    if invalid_cases:
        links = ['<a href="variants/' + escape(item['variant']) + '/' + DATASET + '/validation/results/' + digest(item['sample_id']) + '.json">' + escape(item['variant'] + '/' + item['sample_id']) + '</a>' for item in invalid_cases]
        body += '<p>模型格式失败（无合法Final，部分轨迹未保存）：' + ' · '.join(links) + ' · <a href="verification/">实际人工记录</a> · <a href="validation_gate.json">就绪确认</a></p>'
    body += '<table><thead><tr><th>组</th><th>阶段</th><th>Job IDs</th><th>Validation</th><th>Full</th><th>文件</th></tr></thead><tbody>'
    for variant in VARIANTS:
        spec = plan['variants'][variant]
        directory = Path(spec['root'])
        status = statuses.get(variant) or {'stage': 'validation_pending', 'jobs': {}}
        validation = len(list((directory / DATASET / 'validation/results').glob('*.json')))
        full = len(list((directory / DATASET / 'full').glob('shard_*/results/*.json')))
        jobs = ' · '.join(stage + ': ' + str(job) for stage, job in (status.get('jobs') or {}).items()) or '—'
        link = 'variants/' + variant + '/'
        body += '<tr><td>' + escape(variant) + '</td><td>' + escape(status['stage']) + '</td><td>' + escape(jobs) + '</td><td>' + str(validation) + '/28</td><td>' + str(full) + '/1000</td><td><a href="' + link + 'prompt_diff.patch">Prompt</a> · <a href="' + link + 'logs/">日志</a></td></tr>'
        if status.get('error'):
            body += '<tr><td colspan="6" class="error">' + escape(status['error']) + '</td></tr>'
    body += '</tbody></table><p><a href="plan.json">Plan</a> · <a href="status.json">Status</a> · <a href="code_diff.patch">代码差异</a></p>'
    if plan.get('reuse'):
        body += '<details><summary>各组冻结来源</summary><p class="note">R3/R4/R5 的 validation 和 full 沿用同一旧冻结源码；R1/R2/ALL5 使用新的冻结源码。旧验证仅核验其实际执行版本。</p><table><tr><th>组</th><th>冻结 campaign</th><th>Source hash</th><th>实际输出</th></tr>'
        for variant in VARIANTS:
            spec = plan['variants'][variant]
            version = spec['source_version']
            body += '<tr><td>' + escape(variant) + '</td><td>' + escape(Path(version['campaign_root']).name) + '</td><td><code>' + escape(spec['source_hash'][:16]) + '</code></td><td><a href="variants/' + variant + '/' + DATASET + '/validation/">validation</a> · <a href="variants/' + variant + '/' + DATASET + '/full/">full</a></td></tr>'
        body += '</table><p><a href="source_versions.json">完整来源记录</a></p></details>'
    body += '<p class="note">题数统计已写入的结果文件；有效性由技术 gate 核验。最终报告生成后覆盖本页。</p>'
    if state.get('manual_review_error'):
        body += '<p class="error">人工审查记录待核验：' + escape(state['manual_review_error']) + '</p>'
    css = 'body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;color:#203044;background:#f7f9fc}h1{font-size:24px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px;border-bottom:1px solid #dce4ed;text-align:left}th{background:#eaf0f7}a{color:#175b9c}.note{font-size:13px;color:#536577}.error{color:#a33434;white-space:pre-wrap}'
    atomic_text(root / 'report.zh.html', '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCNB1000 rules-v2 进度</title><style>' + css + '</style><body>' + body + '</body></html>')


def retry_failed(root, variant, stage):
    """Retry identical scientific inputs; completed result files are resumed."""
    root = Path(root).resolve()
    if variant not in VARIANTS or stage not in ('validation', 'full'):
        raise ValueError('Retry requires an approved variant and stage')
    plan = read_json(root / 'plan.json')
    with campaign_locks([root] + plan.get('reuse', {}).get('controller_lock_roots', [])):
        old.verify_frozen(plan)
        state = read_json(root / 'status.json')
        status = state['variants'][variant]
        if status['stage'] != stage + '_failed':
            raise ValueError('Only a failed stage can be retried')
        directory = Path(plan['variants'][variant]['root'])
        receipt = directory / 'receipts' / (DATASET + '-' + stage + '.json')
        if receipt.exists():
            previous = read_json(receipt)
            if old.job_state(previous['job_id'], old.queue_rows(), 10 if stage == 'full' else None) == 'ACTIVE':
                raise ValueError('Previous job remains active; refusing duplicate submission')
            history = directory / 'receipts' / ('attempt-' + stage + '-' + previous['job_id'])
            history.mkdir(parents=True, exist_ok=False)
            for name in (DATASET + '-' + stage + '.json', DATASET + '-' + stage + '-verification.json'):
                path = directory / 'receipts' / name
                if path.exists():
                    shutil.move(str(path), str(history / name))
        status.update(stage=stage + '_pending', last_error=status.pop('error', None))
        status.get('jobs', {}).pop(stage, None)
        state.update(complete=False, updated_at=old.now())
        state.pop('finished_at', None)
        state.pop('gate_error', None)
        atomic_json(root / 'status.json', state)
        old.log(variant + '/' + stage + ': exact-identity retry prepared; restart controller')
        return state


def controller(root, poll_seconds=30):
    root = Path(root).resolve()
    plan = read_json(root / 'plan.json')
    if tuple(plan['variants']) != VARIANTS:
        raise ValueError('Campaign must contain exactly the six approved variants in order')
    with campaign_locks([root] + plan.get('reuse', {}).get('controller_lock_roots', [])):
        state_path = root / 'status.json'
        state = read_json(state_path) if state_path.exists() else {'created_at': old.now(), 'variants': {}}
        state.update(controller_pid=os.getpid(), controller_started_at=old.now())
        last_summary = None
        while True:
            try:
                old.verify_frozen(plan)
            except (ValueError, FileNotFoundError) as exc:
                for status in state['variants'].values():
                    full = status.get('stage', '').startswith('full') or status.get('stage') in FULL_COMPLETED or status.get('jobs', {}).get('full')
                    status.update(stage='full_failed' if full else 'validation_failed', error=str(exc))
                state.update(complete=False, finished_at=old.now(), gate_error='Frozen source/config contract failed: ' + str(exc))
                atomic_json(state_path, state)
                write_pending_html(plan, state)
                return state
            queue = old.queue_rows()
            for variant in VARIANTS:
                status = state['variants'].setdefault(variant, {'stage': 'validation_pending', 'jobs': {}})
                if status['stage'] in (*FULL_COMPLETED, *VALIDATION_READY, 'validation_failed', 'full_failed'):
                    continue
                stage = 'full' if status['stage'].startswith('full') else 'validation'
                try:
                    job = submit_once(plan, variant, stage, queue)
                    if not job:
                        continue
                    status['jobs'][stage] = job
                    queue = old.queue_rows()
                    js = old.job_state(job, queue, 10 if stage == 'full' else None)
                    status.update(stage=stage + '_running', **{stage + '_job_state': js})
                    gate = None
                    if js in old.TERMINAL_FAILURES:
                        if stage == 'validation' and js == 'FAILED':
                            gate = verify_failed_validation_smoke_exit(plan, variant, job)
                        else:
                            raise ValueError('Slurm ' + stage + ' failed: ' + js)
                    if js == 'COMPLETED' or gate is not None:
                        old.verify_frozen(plan)
                        if stage == 'validation':
                            gate = gate or verify_validation(plan, variant, job)
                            gate_path = root / 'verification' / ('validation_' + variant + '.json') if plan.get('controller_update') else Path(plan['variants'][variant]['root']) / 'gate_decision.json'
                            atomic_json(gate_path, gate)
                            status.update(stage=gate['completion_status'], validation_metrics=gate['metrics'])
                            if gate.get('scheduler_failure_acceptance'):
                                status.update(validation_scheduler_failure=gate['scheduler_failure_acceptance'],
                                    validation_exit_code=gate['scheduler_failure_acceptance']['exit_code'],
                                    validation_acceptance_reason=gate['scheduler_failure_acceptance']['reason'])
                        else:
                            metrics = verify_full(plan, variant)
                            status.update(stage=metrics['completion_status'], metrics={DATASET: metrics})
                        old.log(variant + ': ' + status['stage'])
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    status.update(stage=stage + '_failed', error=str(exc))
                    old.log(variant + ': ' + status['stage'] + ': ' + str(exc))
            statuses = state['variants']
            if any(status['stage'] == 'validation_failed' for status in statuses.values()):
                state.update(complete=False, finished_at=old.now(),
                             gate_error='Technical validation failed; no full arrays released')
                atomic_json(state_path, state)
                write_pending_html(plan, state)
                return state
            if release_gate(plan, statuses):
                old.verify_frozen(plan)
                all_valid = all(status['stage'] == 'validation_passed' for status in statuses.values())
                invalid_cases = [{'variant': name, 'sample_id': sid} for name, status in statuses.items() for sid in
                    status.get('validation_metrics', {}).get('execution_contract', {}).get('model_protocol_invalid_sample_ids', [])]
                atomic_json(root / 'technical_validation_gate.json', {'ready': True, 'passed': all_valid,
                    'all_answers_valid': all_valid, 'checked_at': old.now(), 'model_format_invalid_cases': invalid_cases,
                    'arms': list(VARIANTS), 'accuracy_is_gate': False})
                try:
                    manual = manual_validation_gate(plan, statuses)
                    state.pop('manual_review_error', None)
                except (ValueError, KeyError) as exc:
                    manual = None
                    state['manual_review_error'] = str(exc)
                if manual is None:
                    state['manual_validation_review_pending'] = True
                else:
                    atomic_json(root / 'validation_gate.json', {'ready': True, 'passed': all_valid,
                        'all_answers_valid': all_valid, 'checked_at': old.now(), 'model_format_invalid_cases': invalid_cases,
                        'arms': list(VARIANTS), 'accuracy_is_gate': False, 'manual_review': manual})
                    state['manual_validation_review_pending'] = False
                    for variant in VARIANTS:
                        statuses[variant]['stage'] = 'full_pending'
                    old.log('All six fixed validations completed with execution contracts and available-artifact human review; model format failures=' + str(len(invalid_cases)) + '; releasing six arrays of ten shards without GPU cap')
            state.update(updated_at=old.now(), system_queue_tasks=len(queue))
            atomic_json(state_path, state)
            write_pending_html(plan, state)
            summary = {variant: status['stage'] for variant, status in statuses.items()}
            if summary != last_summary:
                old.log('Campaign state: ' + json.dumps(summary))
                last_summary = summary
            if len(statuses) == 6 and all(status['stage'] in (*FULL_COMPLETED, 'full_failed') for status in statuses.values()):
                generate_report(root)
                state.update(complete=all(status['stage'] in FULL_COMPLETED for status in statuses.values()),
                             finished_at=old.now(), report=str(root / 'report.zh.html'))
                atomic_json(state_path, state)
                return state
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'prepare-reuse', 'prepare-controller-update', 'preview', 'run', 'status', 'report', 'retry'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--baseline', default=str(old.BASELINE))
    parser.add_argument('--initial-manifest', default=str(V1 / 'initial_inputs.json'))
    parser.add_argument('--reuse-root')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--stage', choices=('validation', 'full'))
    parser.add_argument('--region-selection-policy', choices=('paper', 'max5'),
                        help='prepare only: new campaigns default to max5; paper reproduces previous counts')
    args = parser.parse_args()
    if args.region_selection_policy is not None and args.mode != 'prepare':
        parser.error('--region-selection-policy is only supported for prepare')
    root = Path(args.root).resolve()
    if args.mode == 'prepare':
        prepare(root, args.baseline, args.initial_manifest,
                region_selection_policy=args.region_selection_policy or 'max5')
    elif args.mode == 'prepare-reuse':
        if not args.reuse_root:
            parser.error('prepare-reuse requires --reuse-root')
        prepare_reuse(root, args.reuse_root)
    elif args.mode == 'prepare-controller-update':
        if not args.reuse_root:
            parser.error('prepare-controller-update requires --reuse-root')
        prepare_controller_update(root, args.reuse_root)
    elif args.mode == 'run':
        if not 10 <= args.poll_seconds <= 60:
            parser.error('poll-seconds must be between 10 and 60')
        controller(root, args.poll_seconds)
    elif args.mode == 'status':
        print(json.dumps(read_json(root / 'status.json'), indent=2))
    elif args.mode == 'retry':
        if not args.variant or not args.stage:
            parser.error('retry requires --variant and --stage')
        retry_failed(root, args.variant, args.stage)
    elif args.mode == 'preview':
        plan = read_json(root / 'plan.json')
        old.verify_frozen(plan)
        for variant in VARIANTS:
            for stage in ('validation', 'full'):
                command = slurm_command(plan, variant, stage)
                old.log(variant + '/' + stage + ': ' + old.run_command(command[:1] + ['--test-only'] + command[1:]).strip())
    else:
        generate_report(root)


if __name__ == '__main__':
    main()
