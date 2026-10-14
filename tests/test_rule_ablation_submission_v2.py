"""Execution invariants for six-arm rules-v2 launch, without GPU or submission."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from slurm import submit_rule_ablations_v2 as launch


def samples():
    return [{'sample_id': 'bcnb:' + str(task * 20 + i), 'task': 'T' + str(task),
             'image_id': str(task * 20 + i // 2), 'ground_truth_label': 'A'}
            for task in range(7) for i in range(12)]


def test_validation_selection_ignores_answers_and_input_order():
    rows = samples()
    selected = launch.validation_samples(rows)
    altered = [{**row, 'ground_truth_label': 'B', 'pred_answer': 'B'} for row in reversed(rows)]
    assert {row['sample_id'] for row in selected} == {row['sample_id'] for row in launch.validation_samples(altered)}
    assert len(selected) == 28
    for task in {row['task'] for row in rows}:
        group = [row for row in selected if row['task'] == task]
        assert len(group) == 4
        assert len({row['image_id'] for row in group}) == 4


def test_commands_have_exact_resources_and_no_baseline_or_array_cap(tmp_path):
    plan = {'account': launch.old.ACCOUNT, 'qos': 'class',
            'variants': {name: {'root': str(tmp_path / name)} for name in launch.VARIANTS}}
    for variant in launch.VARIANTS:
        smoke = launch.slurm_command(plan, variant, 'validation')
        full = launch.slurm_command(plan, variant, 'full')
        assert '--time=480' in smoke
        assert '--time=4320' in full
        assert '--array=0-9' in full
        assert not any('%' in arg for arg in full if arg.startswith('--array'))
        assert '--gres=gpu:rtx_pro_6000:1' in full
        assert '--constraint=rtxpro6000' in full
    with pytest.raises(ValueError):
        launch.slurm_command(plan, 'BASE_COMPAT', 'validation')


def test_release_gate_requires_all_six_validations():
    statuses = {name: {'stage': 'validation_passed'} for name in launch.VARIANTS}
    assert launch.release_gate({}, statuses)
    statuses['ALL5']['stage'] = 'validation_running'
    assert not launch.release_gate({}, statuses)
    statuses.pop('ALL5')
    assert not launch.release_gate({}, statuses)


def trace():
    child = {'region_id': 'p/z40/0_0', 'parent_id': 'p', 'scale': 40}
    finding = {'region': child, 'description': 'cells'}
    return {'process': [
        {'iteration': 1, 'actual_action': 'zoom', 'action_executed': True,
         'requested_target_region': 'p', 'zoom_findings': [finding],
         'decision': {'answer': 'A', 'sufficient': 'No', 'zoom_recommendation': 'Yes', 'zoom_level': 40}},
        {'iteration': 2, 'actual_action': 'conclude', 'action_executed': True,
         'findings': [finding], 'decision': {'answer': 'A', 'sufficient': 'Yes'}}]}


def test_zoom_requires_followup_and_requested_parent():
    result = trace()
    launch.verify_trajectory(result, 'R5_TARGET_REGION')
    wrong = deepcopy(result)
    wrong['process'][0]['requested_target_region'] = 'other'
    with pytest.raises(ValueError, match='target'):
        launch.verify_trajectory(wrong, 'R5_TARGET_REGION')
    wrong = deepcopy(result)
    wrong['process'][1]['findings'] = []
    with pytest.raises(ValueError, match='omitted'):
        launch.verify_trajectory(wrong, 'R5_TARGET_REGION')
    with pytest.raises(ValueError, match='following review'):
        launch.verify_trajectory({'process': result['process'][:1]}, 'R5_TARGET_REGION')


def test_scale_failures_cannot_be_accepted():
    result = trace()
    result['process'][1]['decision']['scale_failures'] = ['nuclei']
    with pytest.raises(ValueError, match='Scale failure'):
        launch.verify_trajectory(result, 'R1_EVIDENCE')
    launch.verify_trajectory(result, 'R5_TARGET_REGION')


def test_last_round_cannot_acquire():
    result = trace()
    result['process'][0]['iteration'] = 5
    with pytest.raises(ValueError, match='following review'):
        launch.verify_trajectory(result, 'ALL5')


def test_r3_checks_authoritative_quality_and_all_invalid_path():
    result = trace()
    result['roi_quality'] = {'p/z40/0_0': {'hard_reject': True}}
    with pytest.raises(ValueError, match='Hard-rejected'):
        launch.verify_trajectory(result, 'R3_ROI_VALIDITY')
    empty = {'process': [], 'stop_reason': 'no_valid_low_regions',
             'total_evidence_regions': 0, 'total_examined_regions': 0, 'initial_region_count': 2,
             'quality_summary': {'checked': 2, 'hard_rejected': 2}}
    launch.verify_trajectory(empty, 'R3_ROI_VALIDITY')
    empty['quality_summary']['hard_rejected'] = 1
    with pytest.raises(ValueError, match='filtering evidence'):
        launch.verify_trajectory(empty, 'R3_ROI_VALIDITY')


def test_manual_gate_requires_real_persisted_review(tmp_path):
    from data_processing.common import atomic_json, digest
    plan = {'root': str(tmp_path), 'variants': {name: {'root': str(tmp_path / name)} for name in launch.VARIANTS}}
    assert launch.manual_validation_gate(plan) is None
    paths = [str(tmp_path / name / launch.DATASET / 'validation') for name in launch.VARIANTS]
    for path in paths:
        Path(path).mkdir(parents=True)
    review = {'complete': True, 'plan_hash': digest(plan), 'reviewed_variants': list(launch.VARIANTS),
              'scope': {'images': True, 'model_outputs': True, 'trajectories': True}, 'reviewed_paths': paths}
    atomic_json(tmp_path / 'verification/manual_validation_review.json', review)
    assert launch.manual_validation_gate(plan)['passed']
    review['reviewed_variants'].pop()
    atomic_json(tmp_path / 'verification/manual_validation_review.json', review)
    assert launch.manual_validation_gate(plan) is None


def test_pending_html_shows_progress_without_accuracy(tmp_path):
    from data_processing.common import atomic_json
    plan = {'root': str(tmp_path),
            'variants': {name: {'root': str(tmp_path / 'variants' / name)} for name in launch.VARIANTS}}
    first = Path(plan['variants'][launch.VARIANTS[0]]['root'])
    atomic_json(first / launch.DATASET / 'validation/results/example.json', {'status': 'ok'})
    atomic_json(first / launch.DATASET / 'full/shard_0/results/example.json', {'status': 'ok'})
    state = {'updated_at': 'now', 'variants': {launch.VARIANTS[0]: {
        'stage': 'validation_running', 'jobs': {'validation': '123'}, 'error': '<test>'}}}
    launch.write_pending_html(plan, state)
    output = (tmp_path / 'report.zh.html').read_text()
    assert '1/28' in output and '1/1000' in output
    assert 'validation: 123' in output
    assert '等待六组技术验证' in output
    assert '等待可用图片、输出及轨迹人工审查' in output
    assert '&lt;test&gt;' in output
    assert all(name in output for name in launch.VARIANTS)
    assert 'accuracy' not in output.lower()
    assert '正确率' not in output


def test_archived_collector_does_not_require_matching_workspace_source(tmp_path, monkeypatch):
    import scripts.run_inference
    from data_processing.common import atomic_json, digest, sha256_file, write_manifest, read_manifest
    from eval.metrics import score
    baseline = tmp_path / 'baseline'
    (baseline / 'source').mkdir(parents=True)
    (baseline / 'source/pathagent.py').write_text('archived entrypoint\n')
    source_hash = digest({'pathagent.py': sha256_file(baseline / 'source/pathagent.py')})
    sample = {'sample_id': 'bcnb:1', 'image_id': '1', 'choices': {'A': 'one', 'B': 'two'},
              'ground_truth_label': 'A', 'dataset': 'bcnb', 'task': 'Tumor'}
    write_manifest(baseline / launch.DATASET / 'full.json', [sample], dataset='bcnb', images_verified=True)
    full = read_manifest(baseline / launch.DATASET / 'full.json')
    config = {key: {} for key in ('models', 'algorithm', 'generation', 'preprocessing', 'runtime')}
    config.update(protocol='old', seed=128)
    identity = {'config': config, 'code_hash': source_hash, 'samples_hash': full['samples_hash'],
                'protocol': 'old', 'prompt_manifest_hash': digest({'templates': {}})}
    run_hash = digest(identity)
    shard = baseline / launch.DATASET / 'full/shard_0'
    atomic_json(shard / 'run_lock.json', {**identity, 'run_hash': run_hash})
    atomic_json(shard / 'prompt_manifest.json', {'templates': {}})
    atomic_json(shard / 'profile.json', launch.FLEET)
    atomic_json(shard / 'image_inventory.json', {'1': {'path': '/unused.jpg', 'sha256': 'original'}})
    atomic_json(shard / 'environment.json', {'executable': sys.executable, 'packages': {}})
    result = {'sample_id': sample['sample_id'], 'run_hash': run_hash, 'status': 'ok', 'pred_answer': 'A'}
    atomic_json(shard / 'results/result.json', result)
    atomic_json(baseline / launch.DATASET / 'metrics.json', score([sample], {sample['sample_id']: result}))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pathagent.py').write_text('new, intentionally incompatible source\n')
    monkeypatch.setattr(launch, 'ROOT', workspace)
    monkeypatch.setattr(scripts.run_inference, 'check_assets', lambda config: None)
    _, history, loaded = launch.archived_baseline(baseline, full, verify_images=False)
    assert history['source_hash'] == source_hash
    assert loaded[sample['sample_id']]['pred_answer'] == 'A'


def test_predecessor_controller_lock_prevents_two_controllers(tmp_path):
    with launch.campaign_locks([tmp_path]):
        with pytest.raises(ValueError, match='Controller remains active'):
            with launch.campaign_locks([tmp_path]):
                pass
    with launch.campaign_locks([tmp_path]):
        pass


def test_reuse_predecessor_requires_receipts_and_no_full_execution(tmp_path):
    from data_processing.common import atomic_json
    plan = {'root': str(tmp_path), 'rule_version': 'rules-v2', 'files': {},
            'variants': {name: {'root': str(tmp_path / name)} for name in launch.VARIANTS}}
    for index, name in enumerate(launch.REUSED_VARIANTS):
        atomic_json(tmp_path / name / 'receipts' / (launch.DATASET + '-validation.json'), {'job_id': str(100 + index)})
    assert set(launch.verify_reuse_predecessor(plan)) == set(launch.REUSED_VARIANTS)
    atomic_json(tmp_path / 'R2_DIFFERENTIAL/receipts' / (launch.DATASET + '-full.json'), {'job_id': '999'})
    with pytest.raises(ValueError, match='full execution'):
        launch.verify_reuse_predecessor(plan)


def test_generic_fingerprint_ignores_decision_only_fix_but_rejects_perceptor_change(tmp_path):
    prior, fresh = tmp_path / 'prior', tmp_path / 'fresh'
    old = 'class RulesV2ModelBackend:\n def describe(self): return "same"\n def evaluate(self): return "old"\n'
    for root in (prior, fresh):
        (root / 'models').mkdir(parents=True)
        (root / 'models/rules_v2_inference.py').write_text(old)
    path = fresh / 'models/rules_v2_inference.py'
    path.write_text(old.replace('"old"', '"new"'))
    assert launch._generic_implementation_identity(prior) == launch._generic_implementation_identity(fresh)
    path.write_text(old.replace('"same"', '"different"'))
    assert launch._generic_implementation_identity(prior) != launch._generic_implementation_identity(fresh)


def test_composed_prepare_retains_three_actual_sources_receipts_and_outputs(tmp_path, monkeypatch):
    from data_processing.common import atomic_json, digest, sha256_file, write_manifest
    previous_root, new_root = tmp_path / 'old', tmp_path / 'new'
    previous_root.mkdir()
    shared = previous_root / 'shared'
    sample = {'sample_id': 'bcnb:1', 'image_id': '1'}
    for name in ('full', 'validation'):
        write_manifest(shared / (name + '.json'), [sample], dataset='bcnb')
    for index in range(10):
        write_manifest(shared / 'shards' / ('shard_' + str(index) + '.json'), [sample], dataset='bcnb')
    previous = {'root': str(previous_root), 'rule_version': 'rules-v2', 'seed': 128,
        'initial_inputs': {'manifest': '/initial', 'sha256': 'fixed'}, 'common_flow': {'zoom_topk': 5},
        'baseline_root': '/baseline', 'files': {}, 'variants': {},
        'datasets': {launch.DATASET: {'full_manifest': str(shared / 'full.json'),
            'validation_manifest': str(shared / 'validation.json'), 'shard_manifests': str(shared / 'shards')}}}
    for index, name in enumerate(launch.VARIANTS):
        directory = previous_root / 'variants' / name
        source = directory / 'source.py'
        source.parent.mkdir(parents=True)
        source.write_text('old ' + name)
        previous['files'][str(source)] = sha256_file(source)
        previous['variants'][name] = {'root': str(directory), 'source': str(source), 'source_hash': 'old-' + name,
            'datasets': {launch.DATASET: {'config_hash': 'old-config'}}}
        atomic_json(directory / 'receipts' / (launch.DATASET + '-validation.json'), {'job_id': str(100 + index)})
    atomic_json(previous_root / 'plan.json', previous)
    original_receipts = {name: (previous_root / 'variants' / name / 'receipts' / (launch.DATASET + '-validation.json')).read_bytes()
                         for name in launch.VARIANTS}
    def fresh_prepare(root, baseline, initial, variants):
        assert variants == launch.FRESH_VARIANTS
        plan = deepcopy(previous)
        plan.update(root=str(root), variants={}, files={})
        for name in variants:
            directory = root / 'variants' / name
            directory.mkdir(parents=True)
            plan['variants'][name] = {'root': str(directory), 'source_hash': 'new-' + name,
                                     'datasets': {launch.DATASET: {'config_hash': 'new-config'}}}
        return plan
    monkeypatch.setattr(launch, 'prepare', fresh_prepare)
    monkeypatch.setattr(launch, 'copy_verified_generic_cache', lambda a, b: {'files': {}, 'records': 0})
    plan = launch.prepare_reuse(new_root, previous_root)
    assert tuple(plan['variants']) == launch.VARIANTS
    for name in launch.REUSED_VARIANTS:
        spec = plan['variants'][name]
        assert spec['root'] == previous['variants'][name]['root']
        assert spec['source_hash'] == 'old-' + name
        assert spec['source_version']['validation_source_hash'] == spec['source_version']['full_source_hash']
        assert (new_root / 'variants' / name).resolve() == Path(spec['root'])
        assert (Path(spec['root']) / 'receipts' / (launch.DATASET + '-validation.json')).read_bytes() == original_receipts[name]
    for name in launch.FRESH_VARIANTS:
        assert plan['variants'][name]['source_hash'] == 'new-' + name
    status = launch.read_json(new_root / 'status.json')
    assert status['variants']['R3_ROI_VALIDITY']['jobs']['validation'] == '102'
    assert status['variants']['R1_EVIDENCE']['jobs'] == {}
    assert status['variants']['R2_DIFFERENTIAL']['jobs'] == {}
    assert all('run_lock' not in path and 'receipts' not in path for path in plan['files'])
    assert '旧验证仅核验其实际执行版本' in (new_root / 'report.zh.html').read_text()


def test_generic_copy_checks_complete_identity_and_excludes_mutable_logs(tmp_path, monkeypatch):
    from data_processing.common import atomic_json, digest, sha256_file
    old_root, new_root = tmp_path / 'old', tmp_path / 'new'
    configs, previous, current = {}, {'variants': {}}, {'variants': {}}
    config = {'seed': 128, 'models': {'perceptor': {'revision': 'pinned'}}, 'generation': {'tokens': 1024},
              'preprocessing': {}, 'algorithm': {}, 'runtime': {}}
    templates = {name: {'text': text, 'sha256': digest(text)} for name, text in
                 [('PERCEPTOR_SYSTEM', 'system'), ('GENERIC_PROMPT', 'H&E'), ('PERCEPTOR_USER', '{metadata}\n{instruction}')]}
    for root, plan in ((old_root, previous), (new_root, current)):
        for name in launch.FRESH_VARIANTS:
            directory = root / name
            key = str(directory / 'config.yaml')
            configs[key] = deepcopy(config)
            configs[key]['runtime']['processed_root'] = str(root / 'cache')
            plan['variants'][name] = {'root': str(directory), 'source': str(directory / 'source'),
                                     'datasets': {launch.DATASET: {'config': key}}}
            atomic_json(directory / 'prompt_manifest.json', {'templates': templates})
    monkeypatch.setattr(launch, 'load_config', lambda path: deepcopy(configs[str(path)]))
    monkeypatch.setattr(launch, '_generic_implementation_identity', lambda source: {'same': 'implementation'})
    resolved = {'temperature': 1.0}
    atomic_json(old_root / 'ALL5' / launch.DATASET / 'validation/resolved_generation.json', {'perceptor': resolved})
    region = {'region_id': 'parent', 'scale': 5}
    geometry = tmp_path / 'geometry'
    atomic_json(geometry / 'regions.json', {'regions': [region]})
    info = {'image_sha256': 'rgb', 'regions_sha256': sha256_file(geometry / 'regions.json'),
            'source_cache': '/original/generic_case', 'directory': str(geometry)}
    initial = tmp_path / 'initial.json'
    atomic_json(initial, {'images': {'1': info}})
    current['initial_inputs'] = {'manifest': str(initial)}
    identity = {'image_sha256': 'rgb', 'geometry_sha256': info['regions_sha256'], 'seed': 128,
        'models': config['models']['perceptor'], 'generation': config['generation'], 'generic_case_id': 'generic_case',
        'prompt_hashes': {name: rec['sha256'] for name, rec in templates.items()},
        'effective_generic_user_template': '{metadata}\nH&E\nInput modality: H&E', 'resolved_perceptor_generation': resolved}
    cache = old_root / 'cache/generic' / digest(identity)
    atomic_json(cache / 'manifest.json', {'image_id': '1', 'identity': identity, 'identity_hash': digest(identity)})
    record = cache / (digest(region) + '.json')
    atomic_json(record, {'identity_hash': digest(identity), 'region': region, 'description': 'Generic only'})
    atomic_json(cache / 'generated_calls_log.json', {'mutable': 'do not copy'})
    value = launch.copy_verified_generic_cache(previous, current)
    assert value['records'] == 1
    copied = list((new_root / 'cache/generic').rglob('*.json'))
    assert [path.name for path in copied] == [record.name]
    assert launch.read_json(copied[0])['description'] == 'Generic only'
    atomic_json(record, {'identity_hash': 'tampered', 'region': region, 'description': 'changed'})
    with pytest.raises(ValueError, match='region identity'):
        launch.copy_verified_generic_cache(previous, current)
