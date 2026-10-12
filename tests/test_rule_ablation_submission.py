"""CPU-only safety and identity checks for the BCNB rule campaign launcher."""
import copy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from data_processing.common import atomic_json, digest, load_config, read_manifest, run_identity, write_manifest
from data_processing.sharding import make_shards
from models.protocols import protocol_manifest
from slurm import submit_rule_ablations as launch


@pytest.fixture
def campaign(tmp_path):
    dataset = 'bcnb_1000_sample'
    variant = 'O_ONLY'
    variant_root = tmp_path / 'variants' / variant
    variant_root.mkdir(parents=True)
    config = load_config('configs/reproduce_v3.yaml')
    config['slurm'].update(shards=10, max_concurrent_gpus=None)
    config_path = variant_root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    sample = {'sample_id': 'bcnb:175', 'image_id': '176', 'dataset': 'bcnb',
              'question': 'Which subtype?', 'choices': {'A': 'one', 'B': 'two'},
              'ground_truth': 'one', 'ground_truth_label': 'A', 'task': 'Tumor', 'source': ''}
    manifest_path = tmp_path / 'smoke.json'
    write_manifest(manifest_path, [sample], dataset='bcnb', purpose='smoke', images_verified=True)
    manifest = read_manifest(manifest_path)
    plan = {'root': str(tmp_path), 'baseline_root': str(tmp_path / 'baseline'),
            'enabled_datasets': [dataset], 'account': 'cs6501-cbx8wm', 'qos': 'class',
            'system_submit_limit': 100, 'maximum_walltime_minutes': 4320, 'files': {},
            'datasets': {dataset: {'smoke_manifest': str(manifest_path), 'shards': 10,
                                  'historical_shard_seconds': [3600.0] * 10,
                                  'baseline_smoke_agent_seconds': {'bcnb:175': 100.0}}},
            'variants': {variant: {'root': str(variant_root), 'source_hash': 'frozen-source',
                                  'prompt_manifest_hash': digest(protocol_manifest(config)),
                                  'datasets': {dataset: {'config': str(config_path),
                                                        'config_hash': digest(config)}}}}}
    identity, run_hash = launch.expected_run(plan, variant, dataset, manifest)
    out = variant_root / dataset / 'smoke'
    atomic_json(out / 'run_lock.json', {**identity, 'run_hash': run_hash})
    fleet = {'gpu': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'gpu_memory_bytes': 101975851008}
    atomic_json(tmp_path / 'gpu_fleet.json', fleet)
    profile = dict(passed=True, real_gpu=True, smoke=True, dataset='bcnb',
                   code_hash=identity['code_hash'], config_hash=digest(config),
                   manifest_hash=manifest['samples_hash'], run_hash=run_hash,
                   slurm_job_id='101', **fleet)
    atomic_json(out / 'profile.json', profile)
    result = dict(sample_id=sample['sample_id'], run_hash=run_hash, status='ok',
                  pred_answer='A', agent_seconds=150.0, process=[])
    result_path = out / 'results' / (digest(sample['sample_id']) + '.json')
    atomic_json(result_path, result)
    generation = {'executor': {'do_sample': True, 'temperature': 0.6},
                  'perceptor': {'do_sample': False}, 'overrides': config['generation'],
                  'image_processor': {'patch_size': 14}, 'context_limit': 40960}
    atomic_json(out / 'resolved_generation.json', generation)
    atomic_json(Path(plan['baseline_root']) / dataset / 'full/shard_0/resolved_generation.json', generation)
    atomic_json(tmp_path / 'plan.json', plan)
    return plan, variant, dataset, out, profile, result_path, result


def test_bcnb_commands_have_exact_shards_one_gpu_and_no_user_throttle(campaign):
    plan, variant, dataset, *_ = campaign
    smoke = launch.slurm_command(plan, variant, dataset, 'smoke')
    full = launch.slurm_command(plan, variant, dataset, 'full', 360)
    assert not any(arg.startswith('--array=') for arg in smoke)
    assert [arg for arg in full if arg.startswith('--array=')] == ['--array=0-9']
    assert '--gres=gpu:rtx_pro_6000:1' in full
    assert '--constraint=rtxpro6000' in full
    assert not any('%' in arg for arg in full if arg.startswith('--array='))
    with pytest.raises(ValueError, match='not enabled'):
        launch.slurm_command(plan, variant, 'cptac', 'smoke')
    with pytest.raises(ValueError, match='not enabled'):
        launch.slurm_command(plan, variant, 'cptac', 'full', 720)


def test_expected_identity_is_the_actual_runner_identity(campaign):
    plan, variant, dataset, *_ = campaign
    manifest = read_manifest(plan['datasets'][dataset]['smoke_manifest'])
    config = load_config(plan['variants'][variant]['datasets'][dataset]['config'])
    identity, run_hash = launch.expected_run(plan, variant, dataset, manifest)
    assert identity == run_identity(config, 'frozen-source', manifest['samples_hash'])
    assert run_hash == digest(identity)


@pytest.mark.parametrize('failure', ['none', 'job', 'run_hash', 'answer', 'generation', 'zoom_float'])
def test_smoke_gate_rejects_stale_or_invalid_evidence(campaign, failure):
    plan, variant, dataset, out, profile, result_path, result = campaign
    if failure == 'job':
        profile['slurm_job_id'] = 'other-job'
        atomic_json(out / 'profile.json', profile)
    elif failure == 'run_hash':
        result['run_hash'] = 'other-run'
    elif failure == 'answer':
        result['pred_answer'] = 'answer outside the choices'
    elif failure == 'generation':
        gen = launch.read_json(out / 'resolved_generation.json')
        gen['perceptor']['do_sample'] = True
        atomic_json(out / 'resolved_generation.json', gen)
    elif failure == 'zoom_float':
        result['process'] = [{'decision': {'zoom_recommendation': 'Yes', 'zoom_level': 20.0}}]
    atomic_json(result_path, result)
    if failure == 'none':
        assert launch.verify_smoke(plan, variant, dataset, '101') == profile
    else:
        with pytest.raises(ValueError):
            launch.verify_smoke(plan, variant, dataset, '101')


def test_shards_keep_all_questions_of_each_wsi_together():
    samples = [{'sample_id': f'bcnb:{i}', 'image_id': str(i // 2)} for i in range(20)]
    shards, _ = make_shards(samples, 10)
    launch.validate_shards(samples, shards, 10)
    broken = copy.deepcopy(shards)
    broken[0][0], broken[1][0] = broken[1][0], broken[0][0]
    with pytest.raises(ValueError, match='image split'):
        launch.validate_shards(samples, broken, 10)


def test_submission_waits_for_system_capacity_and_receipt_prevents_duplicates(campaign, monkeypatch):
    plan, variant, dataset, *_ = campaign
    calls = []
    def command(args):
        calls.append(args)
        return '222\n' if args[0] == 'sbatch' else 'JobId=222'
    monkeypatch.setattr(launch, 'run_command', command)
    queue = [[str(i), 'PENDING', plan['account'], plan['qos']] for i in range(91)]
    assert launch.submit_once(plan, variant, dataset, 'full', queue, 360) is None
    assert calls == []
    assert launch.submit_once(plan, variant, dataset, 'full', queue[:90], 360) == '222'
    assert launch.submit_once(plan, variant, dataset, 'full', queue, 360) == '222'
    assert sum(args[0] == 'sbatch' for args in calls) == 1


@pytest.mark.parametrize('smoke_passes', [True, False])
def test_controller_only_releases_bcnb_after_its_smoke_passes(campaign, monkeypatch, smoke_passes):
    plan, variant, dataset, *_ = campaign
    calls = []
    monkeypatch.setattr(launch, 'queue_rows', lambda: [])
    monkeypatch.setattr(launch, 'verify_frozen', lambda _plan: None)
    monkeypatch.setattr(launch, 'job_state', lambda *args, **kwargs: 'COMPLETED')
    monkeypatch.setattr(launch, 'full_walltime', lambda *args: (360, {}))
    monkeypatch.setattr(launch, 'verify_full', lambda *args: {'complete': True})
    def submit(_plan, arm, ds, stage, queue, minutes=None):
        calls.append((arm, ds, stage))
        return '101' if stage == 'smoke' else '202'
    def verify(*args):
        if not smoke_passes:
            raise ValueError('invalid smoke')
        return {'passed': True}
    monkeypatch.setattr(launch, 'submit_once', submit)
    monkeypatch.setattr(launch, 'verify_smoke', verify)
    state = launch.controller(Path(plan['root']), poll_seconds=0)
    expected = [(variant, dataset, 'smoke')]
    if smoke_passes:
        expected.append((variant, dataset, 'full'))
    assert calls == expected
    assert state['complete'] is smoke_passes


def test_excessive_walltime_stops_instead_of_silently_clamping(campaign):
    plan, variant, dataset, _, _, result_path, result = campaign
    result['agent_seconds'] = 100000.0
    atomic_json(result_path, result)
    with pytest.raises(ValueError, match='partition limit'):
        launch.full_walltime(plan, variant, dataset)


@pytest.mark.parametrize('failed_child', [False, True])
def test_array_accounting_uses_display_ids_not_raw_underlying_job_ids(monkeypatch, failed_child):
    # Real historical arrays have JobID=20637795_0 but JobIDRaw=20637840.
    # The anchor alone can be complete while a different array child failed.
    def command(args):
        final_state = 'FAILED|1:0' if failed_child else 'COMPLETED|0:0'
        if any('JobIDRaw' in arg for arg in args):
            return '20637840|' + final_state + '\n20637795|COMPLETED|0:0\n'
        return '20637795_0|' + final_state + '\n20637795_1|COMPLETED|0:0\n'
    monkeypatch.setattr(launch, 'run_command', command)
    assert launch.job_state('20637795', []) == ('FAILED' if failed_child else 'COMPLETED')


def test_array_requires_all_expected_tasks_before_completion(monkeypatch):
    accounting = ['400_0|COMPLETED|0:0\n400|COMPLETED|0:0\n']
    monkeypatch.setattr(launch, 'run_command', lambda args: accounting[0])
    assert launch.job_state('400', [], task_count=2) == 'UNKNOWN'
    accounting[0] += '400_1|COMPLETED|0:0\n400_1.batch|FAILED|1:0\n'
    assert launch.job_state('400', [], task_count=2) == 'COMPLETED'
    accounting[0] = '400_0|COMPLETED|0:0\n400_1|FAILED|1:0\n'
    assert launch.job_state('400', [], task_count=2) == 'FAILED'


def test_full_evaluation_keeps_invalid_predictions_in_fixed_denominator(campaign):
    plan, variant, dataset, *_ = campaign
    full = Path(plan['variants'][variant]['root']) / dataset / 'full'
    first = read_manifest(plan['datasets'][dataset]['smoke_manifest'])['samples'][0]
    second = {**first, 'sample_id': 'bcnb:879', 'image_id': '880'}
    samples = [first, second]
    manifests = Path(plan['root']) / 'shards'
    full_manifest = Path(plan['root']) / 'full.json'
    write_manifest(full_manifest, samples, dataset='bcnb', purpose='full')
    plan['datasets'][dataset].update(full_manifest=str(full_manifest), shard_manifests=str(manifests), shards=2)
    for i, row in enumerate(samples):
        manifest_path = manifests / f'shard_{i}.json'
        write_manifest(manifest_path, [row], dataset='bcnb', purpose='shard')
        manifest = read_manifest(manifest_path)
        identity, run_hash = launch.expected_run(plan, variant, dataset, manifest)
        atomic_json(full / f'shard_{i}' / 'run_lock.json', {**identity, 'run_hash': run_hash})
        atomic_json(full / f'shard_{i}' / 'results' / (digest(row['sample_id']) + '.json'),
                    {'sample_id': row['sample_id'], 'run_hash': run_hash,
                     'status': 'ok' if i == 0 else 'invalid', 'pred_answer': 'A'})
    metrics = launch.verify_full(plan, variant, dataset)
    assert metrics['complete'] is True
    assert metrics['invalid'] == 1
    assert metrics['closed']['closed_overall'] == {'total': 2, 'correct': 1, 'accuracy_percent': 50.0}


@pytest.mark.parametrize('shards,concurrency', [(6, None), (10, None), (16, 32), (128, 32)])
def test_resource_extension_does_not_relax_generation_or_algorithm(tmp_path, shards, concurrency):
    config = load_config('configs/reproduce_v3.yaml')
    config['slurm'].update(shards=shards, max_concurrent_gpus=concurrency)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)['slurm']['shards'] == shards
    config['algorithm']['initial_ratio'] = 0.2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='frozen paper reproduction'):
        load_config(path)


def test_snapshot_info_imports_the_frozen_prompt_source(tmp_path):
    from slurm.rule_prompts import build_variant_sources
    source = tmp_path / 'source'
    files = ('pathagent.py', 'data_processing/common.py', 'models/protocols.py',
             'models/paper_prompts.py', 'models/context_prompts.py')
    for relative in files:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(launch.ROOT / relative, target)
    for relative, content in build_variant_sources(launch.ROOT, 'F_ONLY').items():
        (source / relative).write_text(content)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(load_config('configs/reproduce_v3.yaml')))
    actual = launch.snapshot_info(source, config_path)
    assert actual['prompts']['variant'] == 'F_ONLY'
    assert actual['prompts']['rules'] == ['F']
    assert len(actual['source_hash']) == 64
