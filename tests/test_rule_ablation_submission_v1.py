"""CPU-only identity, resource and release-gate tests for the v1 study."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from data_processing.common import atomic_json, digest, load_config, read_json, read_manifest, sha256_file, write_manifest
from data_processing.regions import Region
from models.protocols import protocol_manifest
from slurm import submit_rule_ablations_v1 as launch


def call(stage, seed, region=None):
    value = {'stage': stage, 'seed': seed, 'seconds': 1.0,
             'messages': [{'role': 'user', 'content': stage}], 'raw_response': stage,
             'generation': {'do_sample': stage != 'describe'}, 'input_tokens': 10, 'output_tokens': 2}
    if region is not None:
        value.update(region=deepcopy(region), pixel_sha256='original-pixels', prompt='inspect region')
    return value


def baseline_result(row, zoom=False):
    parent = Region('parent', row['image_path'], 0, 0, 4096, 4096, 1024, 1024, 'physical', 5, .5).to_dict()
    child = Region('parent/z40.0/0_0', row['image_path'], 0, 0, 512, 512, 1024, 1024,
                   'physical', 40, .5, parent_id='parent').to_dict()
    event = {'iteration': 1, 'findings': [{'region': parent, 'description': 'low evidence'}],
             'decision': {'answer': 'A', 'thinking_steps': 'morphology', 'sufficient': 'No' if zoom else 'Yes'},
             'action': 'zoom' if zoom else 'conclude', 'examined_region_count': 1}
    calls = [call('describe', 10, parent), call('predict', 11), call('reflect', 12)]
    if zoom:
        event['decision'].update(missing_info='nuclei', zoom_recommendation='Yes', zoom_level=40)
        event.update(zoom_candidate_count=64, zoom_query='nuclei',
                     zoom_finding={'region': child, 'description': 'zoom evidence'})
        calls += [call('missing_info', 13), call('describe', 14, child)]
    calls.append(call('final', 15))
    return {'sample_id': row['sample_id'], 'image_id': row['image_id'], 'status': 'ok',
            'pred_answer': 'A', 'explanation': 'morphology', 'final_response': {'answer': 'A', 'explanation': 'morphology'},
            'stop_reason': 'zoom_completed' if zoom else 'sufficient', 'initial_region_count': 1,
            'total_examined_regions': 1, 'agent_seconds': 10.0, 'process': [event], 'model_calls': calls}


@pytest.fixture
def campaign(tmp_path):
    tasks = ['Tumor', 'Tumor', 'ER', 'PR', 'HER2', 'HER2 Expression',
             'HER2 Expression', 'Histological grading', 'Molecular subtype', 'Molecular subtype']
    rows = [{'sample_id': f'bcnb:{sid}', 'image_id': str(sid + 1), 'dataset': 'bcnb',
             'image_path': f'/unused/{sid + 1}.jpg', 'input_kind': 'wsi', 'assumed_mpp': .5,
             'question': 'Which?', 'choices': {'A': 'one', 'B': 'two'},
             'ground_truth': 'B', 'ground_truth_label': 'B', 'task': task, 'source': ''}
            for sid, task in zip(launch.VALIDATION_IDS, tasks)]
    manifest_path = tmp_path / 'shared/validation.json'
    write_manifest(manifest_path, rows, dataset='bcnb', purpose='smoke', images_verified=True)
    manifest = read_manifest(manifest_path)
    originals = {row['sample_id']: baseline_result(row, row['sample_id'] in {'bcnb:879', 'bcnb:3601', 'bcnb:4573'})
                 for row in rows}
    atomic_json(manifest_path.parent / 'baseline_validation_results.json', originals)
    plan = {'root': str(tmp_path), 'baseline_root': str(tmp_path / 'baseline'),
            'account': 'cs6501-cbx8wm', 'qos': 'class', 'system_submit_limit': 100,
            'maximum_walltime_minutes': 4320, 'files': {}, 'controls': {}, 'variants': {},
            'datasets': {launch.DATASET: {'validation_manifest': str(manifest_path),
                'smoke_manifest': str(manifest_path), 'shards': 10,
                'historical_shard_seconds': [3600.] * 10,
                'baseline_smoke_agent_seconds': {row['sample_id']: 10.0 for row in rows}}}}
    fleet = {'gpu': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'gpu_memory_bytes': 101975851008}
    atomic_json(tmp_path / 'gpu_fleet.json', fleet)
    config = load_config('configs/reproduce_v3.yaml')
    generation = {'executor': {'do_sample': True, 'temperature': .6}, 'perceptor': {'do_sample': False},
                  'overrides': config['generation'], 'image_processor': {'patch_size': 14}, 'context_limit': 40960}
    atomic_json(Path(plan['baseline_root']) / launch.DATASET / 'full/shard_0/resolved_generation.json', generation)
    for variant in (launch.CONTROL,) + launch.VARIANTS:
        directory = tmp_path / ('controls' if variant == launch.CONTROL else 'variants') / variant
        directory.mkdir(parents=True)
        cfg = deepcopy(config)
        cfg['algorithm']['zoom_return_to_loop'] = launch.zoom_enabled(variant)
        cfg['slurm'].update(shards=10, max_concurrent_gpus=None)
        config_path = directory / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        record = {'root': str(directory), 'source_hash': 'source-' + variant,
                  'prompt_manifest_hash': digest(protocol_manifest(cfg)),
                  'datasets': {launch.DATASET: {'config': str(config_path), 'config_hash': digest(cfg)}}}
        plan['controls' if variant == launch.CONTROL else 'variants'][variant] = record
        identity, run_hash = launch.old.expected_run(launch.helper_plan(plan), variant, launch.DATASET, manifest)
        output = directory / launch.DATASET / 'validation'
        atomic_json(output / 'run_lock.json', {**identity, 'run_hash': run_hash})
        atomic_json(output / 'profile.json', {'passed': True, 'real_gpu': True, 'smoke': True, 'dataset': 'bcnb',
            'code_hash': identity['code_hash'], 'config_hash': digest(cfg), 'manifest_hash': manifest['samples_hash'],
            'run_hash': run_hash, 'slurm_job_id': '101', **fleet})
        atomic_json(output / 'resolved_generation.json', generation)
        for sid, original in originals.items():
            result = deepcopy(original)
            result['run_hash'] = run_hash
            if launch.zoom_enabled(variant) and 'zoom_finding' in result['process'][0]:
                first = result['process'][0]
                result['process'].append({'iteration': 2, 'action': 'conclude', 'examined_region_count': 1,
                    'findings': first['findings'] + [first['zoom_finding']],
                    'decision': {'answer': 'A', 'thinking_steps': 'zoom reviewed', 'sufficient': 'Yes'},
                    'requested_action': 'conclude', 'actual_action': 'conclude', 'action_executed': True})
                result['model_calls'][-1:] = [call('predict', 20), call('reflect', 21), call('final', 22)]
                result.update(stop_reason='sufficient', total_zoomed_regions=1, total_evidence_regions=2)
            atomic_json(output / 'results' / (digest(sid) + '.json'), result)
    atomic_json(tmp_path / 'plan.json', plan)
    return plan, originals


def output(plan, variant):
    return Path(launch.all_records(plan)[variant]['root']) / launch.DATASET / 'validation'


def result_path(plan, variant, sid='bcnb:879'):
    return output(plan, variant) / 'results' / (digest(sid) + '.json')


@pytest.mark.parametrize('variant', (launch.CONTROL,) + launch.VARIANTS)
def test_validation_is_one_gpu_and_full_is_ten_tasks_without_percent_cap(campaign, variant):
    plan, _ = campaign
    validation = launch.slurm_command(plan, variant, 'validation')
    assert not any(arg.startswith('--array=') for arg in validation)
    assert '--time=360' in validation
    for command in [validation] + ([] if variant == launch.CONTROL else [launch.slurm_command(plan, variant, 'full', 720)]):
        assert '--partition=gpu' in command
        assert '--gres=gpu:rtx_pro_6000:1' in command
        assert '--constraint=rtxpro6000' in command
        assert '--cpus-per-task=8' in command and '--mem=64G' in command
        assert '--qos=class' in command and '--account=cs6501-cbx8wm' in command
        assert 'cptac' not in ' '.join(command)
        assert not any('%' in arg for arg in command if arg.startswith('--array='))
    if variant == launch.CONTROL:
        with pytest.raises(ValueError):
            launch.slurm_command(plan, variant, 'full')
    else:
        assert '--array=0-9' in launch.slurm_command(plan, variant, 'full', 720)


@pytest.mark.parametrize('variant,minimum', [(v, 720 if launch.zoom_enabled(v) else 360) for v in launch.VARIANTS])
def test_full_walltime_enforces_zoom_minimum_and_never_clamps(campaign, variant, minimum):
    plan, _ = campaign
    minutes, _ = launch.full_walltime(plan, variant)
    assert minutes == minimum
    assert '--time=' + str(minimum) in launch.slurm_command(plan, variant, 'full', minutes)
    plan['datasets'][launch.DATASET]['historical_shard_seconds'] = [1e8]
    with pytest.raises(ValueError, match='partition limit'):
        launch.full_walltime(plan, variant)


def test_submit_capacity_and_receipts_prevent_duplicate_arrays(campaign, monkeypatch):
    plan, _ = campaign
    calls = []
    def command(args):
        calls.append(args)
        return '222;cluster\n' if args[0] == 'sbatch' else 'JobId=222'
    monkeypatch.setattr(launch.old, 'run_command', command)
    queue = [[str(i), 'PENDING', plan['account'], plan['qos']] for i in range(91)]
    assert launch.submit_once(plan, 'C_ZOOM_LOOP', 'full', queue, 720) is None
    assert calls == []
    assert launch.submit_once(plan, 'C_ZOOM_LOOP', 'full', queue[:90], 720) == '222'
    assert launch.submit_once(plan, 'C_ZOOM_LOOP', 'full', queue, 720) == '222'
    assert sum(c[0] == 'sbatch' for c in calls) == 1


@pytest.mark.parametrize('variant', (launch.CONTROL,) + launch.VARIANTS)
def test_valid_technical_gate_does_not_require_accuracy(campaign, variant):
    plan, _ = campaign
    gate = launch.verify_validation(plan, variant, '101')
    assert gate['passed'] and not gate['accuracy_is_gate']
    assert gate['metrics']['closed']['closed_overall']['correct'] == 0


@pytest.mark.parametrize('failure', ['job', 'run_hash', 'answer', 'generation', 'zoom_float'])
def test_validation_rejects_stale_or_invalid_outputs(campaign, failure):
    plan, _ = campaign
    variant = 'A_HE_PRED'
    directory = output(plan, variant)
    if failure == 'job':
        profile = read_json(directory / 'profile.json')
        profile['slurm_job_id'] = 'other'
        atomic_json(directory / 'profile.json', profile)
    elif failure == 'generation':
        gen = read_json(directory / 'resolved_generation.json')
        gen['perceptor']['do_sample'] = True
        atomic_json(directory / 'resolved_generation.json', gen)
    else:
        path = result_path(plan, variant)
        result = read_json(path)
        if failure == 'run_hash':
            result['run_hash'] = 'other'
        elif failure == 'answer':
            result['pred_answer'] = 'Unknown'
        else:
            result['process'][0]['decision']['zoom_level'] = 40.0
        atomic_json(path, result)
    with pytest.raises(ValueError):
        launch.verify_validation(plan, variant, '101')


@pytest.mark.parametrize('field', ['seed', 'messages', 'raw_response', 'generation', 'pixel_sha256', 'region', 'stage'])
def test_stable_trace_keeps_scientific_call_fields(campaign, field):
    _, originals = campaign
    before = originals['bcnb:879']
    after = deepcopy(before)
    after['model_calls'][0][field] = 'changed'
    assert launch.stable_trace(before) != launch.stable_trace(after)


def test_stable_trace_ignores_only_expected_bookkeeping_and_timing(campaign):
    _, originals = campaign
    before = originals['bcnb:879']
    after = deepcopy(before)
    after['agent_seconds'] = 999
    after['model_calls'][0]['seconds'] = 999
    for key in launch.EVENT_ADDITIONS:
        after['process'][0][key] = 'bookkeeping'
    after.update(total_zoomed_regions=1, total_evidence_regions=2)
    assert launch.stable_trace(before) == launch.stable_trace(after)
    after['process'][0]['decision']['sufficient'] = 'Yes'
    assert launch.stable_trace(before) != launch.stable_trace(after)


@pytest.mark.parametrize('variant', [launch.CONTROL, 'C_ZOOM_LOOP'])
def test_baseline_compat_and_c_no_zoom_reject_changed_seed(campaign, variant):
    plan, _ = campaign
    path = result_path(plan, variant, 'bcnb:336')
    result = read_json(path)
    result['model_calls'][0]['seed'] += 1
    atomic_json(path, result)
    with pytest.raises(ValueError, match='differ|changed'):
        launch.verify_validation(plan, variant, '101')


@pytest.mark.parametrize('failure', ['prefix_seed', 'prefix_message', 'no_review', 'no_reflect', 'reflect_before_predict', 'final_before_reflect', 'child_missing'])
def test_c_gate_requires_unchanged_prefix_and_actual_post_zoom_review(campaign, failure):
    plan, _ = campaign
    path = result_path(plan, 'C_ZOOM_LOOP')
    result = read_json(path)
    if failure == 'prefix_seed':
        result['model_calls'][1]['seed'] += 1
    elif failure == 'prefix_message':
        result['model_calls'][1]['messages'][0]['content'] = 'changed'
    elif failure == 'child_missing':
        result['process'][1]['findings'].pop()
    else:
        prefix = result['model_calls'][:5]
        suffix = {'no_review': ['final'], 'no_reflect': ['predict', 'final'],
                  'reflect_before_predict': ['reflect', 'predict', 'final'],
                  'final_before_reflect': ['predict', 'final', 'reflect']}[failure]
        result['model_calls'] = prefix + [call(stage, 30 + i) for i, stage in enumerate(suffix)]
    atomic_json(path, result)
    with pytest.raises(ValueError):
        launch.verify_validation(plan, 'C_ZOOM_LOOP', '101')


def test_c_gate_allows_predict_format_retry_before_reflection(campaign):
    plan, _ = campaign
    path = result_path(plan, 'C_ZOOM_LOOP')
    result = read_json(path)
    result['model_calls'].insert(6, call('predict', 100))
    atomic_json(path, result)
    assert launch.verify_validation(plan, 'C_ZOOM_LOOP', '101')['passed']


@pytest.mark.parametrize('summary_stages', [('predict',), ('reflect',), ('predict', 'reflect')])
def test_c_gate_allows_evidence_summaries_around_post_zoom_review(campaign, summary_stages):
    plan, _ = campaign
    path = result_path(plan, 'C_ZOOM_LOOP')
    result = read_json(path)
    prefix, review = result['model_calls'][:5], result['model_calls'][5:]
    expanded = []
    for item in review:
        if item['stage'] in summary_stages:
            expanded.append(call('summary', 100 + len(expanded)))
        expanded.append(item)
    result['model_calls'] = prefix + expanded
    atomic_json(path, result)
    assert launch.verify_validation(plan, 'C_ZOOM_LOOP', '101')['passed']


@pytest.mark.parametrize('failed_arm', [None, launch.CONTROL, 'ALL4'])
def test_controller_waits_for_all_six_validations_before_any_full_array(campaign, monkeypatch, failed_arm):
    plan, _ = campaign
    verified, submissions, full_checks = set(), [], []
    monkeypatch.setattr(launch.old, 'verify_frozen', lambda _: None)
    monkeypatch.setattr(launch.old, 'queue_rows', lambda: [])
    def job_state(job, queue, task_count=None):
        if job.startswith('full-'):
            full_checks.append(task_count)
        return 'COMPLETED'
    monkeypatch.setattr(launch.old, 'job_state', job_state)
    monkeypatch.setattr(launch.old, 'verify_full', lambda *args: {'complete': True})
    def submit(_plan, variant, stage, queue, minutes=None):
        if stage == 'full':
            assert verified == set(launch.all_records(plan))
            assert read_json(Path(plan['root']) / 'validation_gate.json')['passed']
            assert variant != launch.CONTROL
        submissions.append((variant, stage))
        return stage + '-' + variant
    def verify(_plan, variant, job):
        if variant == failed_arm:
            raise ValueError('test validation failure')
        verified.add(variant)
        return {'passed': True, 'metrics': {'complete': True}}
    monkeypatch.setattr(launch, 'submit_once', submit)
    monkeypatch.setattr(launch, 'verify_validation', verify)
    monkeypatch.setattr(launch.time, 'sleep', lambda _: None)
    monkeypatch.setattr(launch.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    state = launch.controller(Path(plan['root']), poll_seconds=0)
    full_submissions = [v for v, stage in submissions if stage == 'full']
    assert state['complete'] is (failed_arm is None)
    assert full_submissions == (list(launch.VARIANTS) if failed_arm is None else [])
    assert full_checks == ([10] * 5 if failed_arm is None else [])


def test_initial_input_builder_copies_generic_provenance_and_sample_contract(tmp_path):
    config = load_config('configs/reproduce_v3.yaml')
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'frozen-image')
    image_sha = sha256_file(image)
    source = tmp_path / 'cache'
    source.mkdir()
    baseline = tmp_path / 'baseline'
    atomic_json(baseline / launch.DATASET / 'full/shard_0/run_lock.json', {'code_hash': 'old-source'})
    row = {'sample_id': 'bcnb:one', 'image_id': 'one', 'dataset': 'bcnb', 'input_kind': 'wsi', 'assumed_mpp': .5}
    key = digest({'image': image_sha, 'image_path': str(image), 'kind': row['input_kind'], 'mpp': row['assumed_mpp'],
                  'preprocessing': config['preprocessing'], 'initial_magnification': 5, 'code': 'old-source'})
    region = Region('one', str(image.resolve()), 0, 0, 4096, 4096, 1024, 1024, 'physical', 5, .5)
    atomic_json(source / 'regions.json', {'key': key, 'image_sha256': image_sha, 'regions': [region.to_dict()]})
    obs_key = digest({'models': config['models'], 'generation': config['generation'],
                      'seed': config['seed'], 'protocol': protocol_manifest(config)})
    observations = {'descriptions': {'one': 'old generic'}, 'features': {'one': [1., 0.]}, 'seconds': 10}
    atomic_json(source / ('observations_' + obs_key + '.json'), observations)
    partial = source / ('description_' + digest([obs_key, 'one']) + '.json')
    atomic_json(partial, {'description': 'old generic', 'model_call': call('describe', 128, region.to_dict())})
    history = {'images': {'one': {'observation_cache': str(source), 'path': str(image), 'sha256': image_sha,
                                  'width': 4096, 'height': 4096}}}
    summary = launch.freeze_initial_inputs(tmp_path / 'new', baseline, config, history,
                                          {'samples_hash': 'samples', 'samples': [row]})
    manifest = read_json(summary['manifest'])
    frozen = manifest['images']['one']
    assert {key: frozen[key] for key in ('dataset', 'input_kind', 'assumed_mpp')} == {
        key: row[key] for key in ('dataset', 'input_kind', 'assumed_mpp')}
    assert frozen['image_path'] == str(image.resolve())
    assert summary['generic_fresh_calls'] == 0 and summary['generic_model_calls_copied'] == 1
    assert (Path(frozen['directory']) / partial.name).read_bytes() == partial.read_bytes()
    assert read_json(Path(frozen['directory']) / 'observations.json') == observations
    assert sha256_file(Path(frozen['directory']) / 'regions.json') == frozen['regions_sha256']
    assert sha256_file(summary['manifest']) == summary['sha256']
