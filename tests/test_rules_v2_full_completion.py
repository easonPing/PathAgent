"""Full-run model errors keep the denominator; execution failures never pass."""
from copy import deepcopy
from pathlib import Path

import pytest

from data_processing.common import atomic_json, digest, read_json, sha256_file, write_manifest
from data_processing.regions import Region, zoom_regions
from eval.metrics import score
from slurm import report_rule_ablations_v2 as report
from slurm import submit_rule_ablations_v2 as launch


def ok(sid='1'):
    return {'sample_id': sid, 'image_id': sid, 'status': 'ok', 'pred_answer': 'A',
            'process': [{'iteration': 1, 'actual_action': 'conclude', 'action_executed': True,
                         'decision': {'answer': 'A', 'sufficient': 'Yes'}}]}


def invalid(sid='1', stage='final'):
    return {'sample_id': sid, 'image_id': sid, 'status': 'invalid', 'pred_answer': '',
            'error': f'Invalid {stage} response after 2 attempts', 'error_stage': stage,
            'model_calls': [{'stage': stage, 'raw_response': '{bad first'},
                            {'stage': stage, 'raw_response': '{bad retry'}]}


def test_full_1000_format_error_preserves_denominator_and_raw(tmp_path, monkeypatch):
    samples = [{'sample_id': str(i), 'image_id': str(i), 'choices': {'A': 'one', 'B': 'two'},
                'ground_truth_label': 'A'} for i in range(1000)]
    results = {row['sample_id']: ok(row['sample_id']) for row in samples}
    results['999'] = invalid('999')
    original = deepcopy(results)
    manifest = tmp_path / 'full.json'
    write_manifest(manifest, samples, dataset='bcnb')
    plan = {'variants': {'R2_DIFFERENTIAL': {'root': str(tmp_path / 'arm')}},
            'datasets': {launch.DATASET: {'full_manifest': str(manifest)}}}
    monkeypatch.setattr(launch.old, 'verify_full', lambda *args: score(samples, results))
    monkeypatch.setattr(launch.old, 'load_results', lambda *args: results)
    metrics = launch.verify_full(plan, 'R2_DIFFERENTIAL')
    assert metrics['expected'] == metrics['received'] == 1000
    assert metrics['invalid'] == 1
    assert metrics['closed']['closed_overall']['total'] == 1000
    assert metrics['closed']['closed_overall']['accuracy_percent'] == 99.9
    assert metrics['completion_status'] == 'completed_with_model_errors'
    assert metrics['execution_contract']['invalid_without_serialized_partial_trajectory'] == ['999']
    strict = report.old.group_metrics(samples, results)
    assert strict['expected'] == 1000 and strict['invalid'] == 1 and strict['correct'] == 999
    assert results == original  # No raw response, class or evidence is filled in.


@pytest.mark.parametrize('error', [
    'Context overflow at final: 30000 + 512', 'Conflicting metadata for Region p',
    'Requested Zoom parent is not eligible', 'Unsupported Zoom level',
    'Missing an executable information request', 'CUDA out of memory', 'I/O read failed'])
def test_non_format_failure_is_never_completed_model_error(error):
    result = invalid()
    result['error'] = error
    with pytest.raises(ValueError, match='Non-format'):
        launch.audit_full_results({}, 'R2_DIFFERENTIAL', {'1': result})


def test_format_error_requires_actual_matching_failed_calls():
    result = invalid()
    result['model_calls'][-1]['stage'] = 'predict'
    assert not launch.model_format_invalid(result)
    with pytest.raises(ValueError):
        launch.audit_full_results({}, 'R2_DIFFERENTIAL', {'1': result})


def test_model_error_does_not_hide_another_questions_budget_violation():
    result = ok('2')
    result['process'] *= 6
    with pytest.raises(ValueError, match='budget'):
        launch.audit_full_results({}, 'R2_DIFFERENTIAL', {'1': invalid(), '2': result})


def test_actual_child_coordinates_are_checked_even_after_model_format_error():
    parent = Region('p', '/image.jpg', 0, 0, 4096, 4096, 1024, 1024,
                    'physical', 5.0, 0.5, 'assumed')
    child = zoom_regions(parent, 40.0)[0].to_dict()
    result = invalid()
    result['model_calls'].insert(0, {'stage': 'describe', 'region': parent.to_dict(), 'raw_response': 'tissue'})
    result['model_calls'].insert(1, {'stage': 'describe', 'region': child, 'raw_response': 'detail'})
    launch.verify_result_geometry(result, {'p': parent.to_dict()})
    result['model_calls'][1]['region']['x'] += 1
    with pytest.raises(ValueError, match='coordinates'):
        launch.audit_full_results({}, 'R2_DIFFERENTIAL', {'1': result})


@pytest.mark.parametrize('illegal_parent', [False, True])
def test_partial_format_failure_cannot_hide_illegal_acquisition_scale(illegal_parent):
    parent = Region('p', '/image.jpg', 0, 0, 4096, 4096, 1024, 1024,
                    'physical', 10.0 if illegal_parent else 5.0, 0.5, 'assumed')
    result = invalid()
    result['model_calls'].insert(0, {'stage': 'describe', 'region': parent.to_dict(),
                                     'region_metadata': parent.metadata(), 'raw_response': 'tissue'})
    if not illegal_parent:
        child = zoom_regions(parent, 80.0)[0]
        # The failed Perceptor has only raw acquisition metadata, matching actual3033's storage shape.
        result['model_calls'].insert(1, {'stage': 'describe', 'region_metadata': child.metadata(),
                                         'raw_response': 'failed detail'})
    with pytest.raises(ValueError, match='Illegal acquired .* magnification'):
        launch.audit_full_results({}, 'R2_DIFFERENTIAL', {'1': result})


def report_fixture(tmp_path):
    samples = [{'sample_id': str(i), 'image_id': str(i), 'choices': {'A': 'one', 'B': 'two'},
                'ground_truth_label': 'A', 'question': 'Which?', 'task': 'Tumor'} for i in range(2)]
    full = tmp_path / 'full.json'
    write_manifest(full, samples, dataset='bcnb')
    manifest = read_json(full)
    plan = {'root': str(tmp_path), 'baseline_root': str(tmp_path / 'BASELINE'), 'files': {},
            'datasets': {report.DATASET: {'full_manifest': str(full), 'baseline_expected_correct': 2}},
            'variants': {}}
    for name in ('BASELINE',) + report.FAMILY:
        directory = tmp_path / name / report.DATASET
        write_manifest(directory / 'manifests/shard_0.json', samples, dataset='bcnb')
        config, prompt = {'protocol': 'fixture'}, {'templates': {}}
        identity = {'config': config, 'code_hash': 'source', 'samples_hash': manifest['samples_hash'],
                    'protocol': 'fixture', 'prompt_manifest_hash': digest(prompt)}
        run_hash = digest(identity)
        atomic_json(directory / 'full/shard_0/run_lock.json', {**identity, 'run_hash': run_hash})
        for row in samples:
            value = ok(row['sample_id'])
            if name == 'R2_DIFFERENTIAL' and row['sample_id'] == '1':
                value = invalid('1')
            value['run_hash'] = run_hash
            atomic_json(directory / 'full/shard_0/results' / (digest(row['sample_id']) + '.json'), value)
        if name != 'BASELINE':
            plan['variants'][name] = {'root': str(tmp_path / name), 'source_hash': 'source',
                'prompt_manifest_hash': digest(prompt), 'datasets': {report.DATASET: {'config_hash': digest(config)}}}
    atomic_json(tmp_path / 'plan.json', plan)
    atomic_json(tmp_path / 'status.json', {'variants': {name: {
        'stage': 'completed_with_model_errors' if name == 'R2_DIFFERENTIAL' else 'complete'} for name in report.FAMILY}})
    return plan


def test_report_completed_model_errors_explicitly_lists_invalid(tmp_path):
    report_fixture(tmp_path)
    values = report.build_report(tmp_path, iterations=16)
    data = values[0]
    assert data['complete'] and data['data_complete'] and data['execution_contract_passed']
    assert data['status'] == 'completed-with-model-errors'
    metric = data['arms']['R2_DIFFERENTIAL']['metrics']['overall']
    assert metric['expected'] == 2 and metric['invalid'] == 1 and metric['accuracy_percent'] == 50
    report.write_report(tmp_path, *values)
    page = (tmp_path / 'report.zh.html').read_text()
    assert 'Invalid / Missing' in page and 'completed-with-model-errors' in page


@pytest.mark.parametrize('failure', ['controller', 'budget', 'source'])
def test_report_never_claims_complete_for_contract_failure(tmp_path, failure):
    plan = report_fixture(tmp_path)
    if failure == 'controller':
        state = read_json(tmp_path / 'status.json')
        state['variants']['R2_DIFFERENTIAL'].update(stage='full_failed', error='Target mismatch')
        atomic_json(tmp_path / 'status.json', state)
    elif failure == 'budget':
        path = tmp_path / 'R2_DIFFERENTIAL' / report.DATASET / 'full/shard_0/results' / (digest('0') + '.json')
        result = read_json(path)
        result['process'] *= 6
        atomic_json(path, result)
    else:
        source = tmp_path / 'frozen.py'
        source.write_text('original')
        plan['files'][str(source)] = sha256_file(source)
        atomic_json(tmp_path / 'plan.json', plan)
        source.write_text('changed')
    data = report.build_report(tmp_path, iterations=16)[0]
    assert data['data_complete'] and not data['complete']
    assert data['status'] == 'full_failed'
    assert data['arms']['R2_DIFFERENTIAL']['execution_contract']['passed'] is False
    assert data['arms']['R2_DIFFERENTIAL']['paired_with_baseline']['overall']['bootstrap']['status'] == 'withheld_execution_contract'


def controller_predecessor(tmp_path):
    previous, ancestor, live = tmp_path / 'previous', tmp_path / 'ancestor', tmp_path / 'live'
    previous.mkdir(); ancestor.mkdir()
    (previous / 'controller_source/slurm').mkdir(parents=True)
    (live / 'slurm').mkdir(parents=True)
    files = {}
    for name in ('submit_rule_ablations_v2.py', 'report_rule_ablations_v2.py'):
        frozen = previous / 'controller_source/slurm' / name
        frozen.write_text('old controller')
        (live / 'slurm' / name).write_text('new controller')
        files[str(frozen)] = sha256_file(frozen)
    specs = {}
    for i, name in enumerate(launch.VARIANTS):
        directory = previous / 'variants' / name
        source = directory / 'source/model.py'
        source.parent.mkdir(parents=True)
        source.write_text('untouched actual model ' + name)
        files[str(source)] = sha256_file(source)
        specs[name] = {'root': str(directory), 'source': str(source.parent), 'source_hash': 'old-' + name,
                      'source_version': {'campaign_root': str(previous)},
                      'datasets': {launch.DATASET: {'config': '/unchanged.yaml', 'config_hash': 'fixed'}}}
        atomic_json(directory / 'receipts' / (launch.DATASET + '-validation.json'), {'job_id': str(100 + i)})
        atomic_json(directory / launch.DATASET / 'validation/run_lock.json', {'real_previous_lock': name})
        atomic_json(directory / launch.DATASET / 'validation/results/result.json', {'real_previous_result': name})
    atomic_json(previous / 'verification/manual_R2.json', {'actual_human_review': True})
    plan = {'root': str(previous), 'rule_version': 'rules-v2', 'variants': specs, 'files': files,
            'reuse': {'controller_lock_roots': [str(ancestor)]}}
    atomic_json(previous / 'plan.json', plan)
    return previous, ancestor, live, plan


def test_controller_update_reuses_all_actual_model_specs_and_results(tmp_path, monkeypatch):
    previous, ancestor, live, before = controller_predecessor(tmp_path)
    paths = [path for path in previous.rglob('*.json') if 'validation' in str(path)]
    original = {str(path): path.read_bytes() for path in paths}
    monkeypatch.setattr(launch, 'ROOT', live)
    new = tmp_path / 'new'
    plan = launch.prepare_controller_update(new, previous)
    assert plan['variants'] == before['variants']
    assert set(plan['reuse']['controller_lock_roots']) == {str(previous), str(ancestor)}
    assert set(plan['reuse']['validation_jobs']) == set(launch.VARIANTS)
    for name in launch.VARIANTS:
        assert (new / 'variants' / name).resolve() == Path(before['variants'][name]['root'])
    assert original == {str(path): path.read_bytes() for path in paths}
    assert (new / 'verification/manual_R2.json').is_symlink()
    assert (new / 'controller_source/slurm/submit_rule_ablations_v2.py').read_text() == 'new controller'
    assert all(value['jobs']['validation'].isdigit() for value in read_json(new / 'status.json')['variants'].values())
    assert not any('receipts' in path or 'run_lock' in path or '/results/' in path for path in plan['files'])


def test_controller_update_refuses_existing_full_receipt_without_touching_it(tmp_path, monkeypatch):
    previous, ancestor, live, plan = controller_predecessor(tmp_path)
    receipt = previous / 'variants/R2_DIFFERENTIAL/receipts' / (launch.DATASET + '-full.json')
    atomic_json(receipt, {'job_id': 'existing-real-job'})
    original = receipt.read_bytes()
    monkeypatch.setattr(launch, 'ROOT', live)
    with pytest.raises(ValueError, match='full execution'):
        launch.prepare_controller_update(tmp_path / 'new', previous)
    assert receipt.read_bytes() == original and not (tmp_path / 'new').exists()


def test_controller_all_model_error_completions_are_terminal(tmp_path, monkeypatch):
    plan = {'root': str(tmp_path), 'variants': {name: {'root': str(tmp_path / name)}
            for name in launch.VARIANTS}, 'files': {}}
    atomic_json(tmp_path / 'plan.json', plan)
    atomic_json(tmp_path / 'status.json', {'variants': {name: {'stage': 'full_pending', 'jobs': {}}
                                                       for name in launch.VARIANTS}})
    monkeypatch.setattr(launch.old, 'queue_rows', lambda: [])
    monkeypatch.setattr(launch, 'submit_once', lambda *args: '100')
    monkeypatch.setattr(launch.old, 'job_state', lambda *args: 'COMPLETED')
    monkeypatch.setattr(launch, 'verify_full', lambda *args: {'completion_status': 'completed_with_model_errors'})
    monkeypatch.setattr(launch, 'generate_report', lambda root: None)
    state = launch.controller(tmp_path)
    assert state['complete'] is True
    assert all(value['stage'] == 'completed_with_model_errors' for value in state['variants'].values())


def test_controller_source_drift_persists_full_failed_without_submitting(tmp_path, monkeypatch):
    source = tmp_path / 'frozen.py'
    source.write_text('original')
    plan = {'root': str(tmp_path), 'variants': {name: {'root': str(tmp_path / name)}
            for name in launch.VARIANTS}, 'files': {str(source): sha256_file(source)}}
    atomic_json(tmp_path / 'plan.json', plan)
    atomic_json(tmp_path / 'status.json', {'variants': {name: {'stage': 'full_pending', 'jobs': {}}
                                                       for name in launch.VARIANTS}})
    source.write_text('drifted')
    monkeypatch.setattr(launch.old, 'queue_rows', lambda: pytest.fail('No jobs after source drift'))
    state = launch.controller(tmp_path)
    assert state['complete'] is False
    assert all(value['stage'] == 'full_failed' for value in state['variants'].values())
    assert 'Frozen source/config contract failed' in state['gate_error']


def validation_fixture(tmp_path, monkeypatch, result):
    variant = 'ALL5'
    row = {'sample_id': '1', 'image_id': '1', 'choices': {'A': 'one', 'B': 'two'}, 'ground_truth_label': 'A'}
    manifest = {'dataset': 'bcnb', 'samples': [row], 'samples_hash': digest([row])}
    baseline = tmp_path / 'baseline'
    plan = {'variants': {variant: {'root': str(tmp_path)}}, 'baseline_root': str(baseline),
            'datasets': {launch.DATASET: {'validation_manifest': '/unused'}}}
    identity, run_hash = {'code_hash': 'actual-source', 'config': {}}, 'actual-run'
    output = tmp_path / launch.DATASET / 'validation'
    atomic_json(output / 'run_lock.json', {**identity, 'run_hash': run_hash})
    atomic_json(output / 'profile.json', {'passed': False, 'real_gpu': True, 'smoke': True,
        'dataset': 'bcnb', 'code_hash': 'actual-source', 'config_hash': digest({}),
        'manifest_hash': manifest['samples_hash'], 'run_hash': run_hash, 'slurm_job_id': '123', **launch.FLEET})
    generation = {'executor': {'frozen': True}, 'perceptor': {'frozen': True},
                  'overrides': {}, 'image_processor': {}, 'context_limit': 40960}
    atomic_json(output / 'resolved_generation.json', generation)
    atomic_json(baseline / launch.DATASET / 'full/shard_0/resolved_generation.json', generation)
    result['run_hash'] = run_hash
    atomic_json(output / 'results' / (digest('1') + '.json'), result)
    monkeypatch.setattr(launch, 'read_manifest', lambda *args: manifest)
    monkeypatch.setattr(launch.old, 'expected_run', lambda *args: (identity, run_hash))
    return plan, variant


def test_validation_readiness_keeps_capped_perceptor_failure_and_absent_partial_process(tmp_path, monkeypatch):
    result = invalid(stage='describe')
    result['error'] = 'Invalid tissue-status Question description after two attempts'
    parent = Region('p', '/image.jpg', 32768, 20480, 4096, 4096, 1024, 1024,
                    'physical', 5.0, 0.5, 'assumed')
    child = zoom_regions(parent, 20.0)[6]
    result['model_calls'] = [
        {'stage': 'describe', 'region': parent.to_dict(), 'region_metadata': parent.metadata(), 'raw_response': 'visible tissue'},
        *[{'stage': 'describe', 'region_metadata': child.metadata(), 'raw_response': '{"tissue_status":"present","findings":"PR_1/2/3/',
           'output_tokens': 2048, 'at_token_cap': True, 'format_attempt': attempt}
          for attempt in range(2)]]
    plan, variant = validation_fixture(tmp_path, monkeypatch, result)
    gate = launch.verify_validation(plan, variant, '123')
    assert gate['ready'] and not gate['passed'] and not gate['all_answers_valid']
    assert gate['completion_status'] == 'validation_completed_with_model_errors'
    assert gate['metrics']['expected'] == gate['metrics']['received'] == 1
    assert gate['metrics']['invalid'] == 1
    assert gate['trajectory_checks']['successful_questions'] == 0
    assert gate['trajectory_checks']['invalid_without_serialized_partial_trajectory'] == ['1']
    assert 'process' not in result and 'final_response' not in result


def test_validation_unknown_failure_still_blocks(tmp_path, monkeypatch):
    result = invalid()
    result['error'] = 'Context overflow at final: 45000 + 2048'
    plan, variant = validation_fixture(tmp_path, monkeypatch, result)
    with pytest.raises(ValueError, match='Non-format'):
        launch.verify_validation(plan, variant, '123')


@pytest.mark.parametrize('stage', ['predict', 'reflect', 'missing_info', 'final'])
def test_raw_calls_block_provable_retry_budget_violation(stage):
    result = invalid()
    result['model_calls'][:0] = [{'stage': stage, 'raw_response': 'raw'}] * 3
    with pytest.raises(ValueError, match='retry contract'):
        launch.audit_full_results({}, 'ALL5', {'1': result})


def test_failed_perceptor_metadata_and_successful_raw_metadata_cannot_drift():
    parent = Region('p', '/image.jpg', 0, 0, 4096, 4096, 1024, 1024,
                    'physical', 5.0, 0.5, 'assumed')
    child = zoom_regions(parent, 20.0)[0]
    result = invalid()
    result['model_calls'][:0] = [{'stage': 'describe', 'region_metadata': child.metadata(), 'raw_response': 'failed'}]
    observed = launch.verify_result_geometry(result, {'p': parent.to_dict()})
    assert observed[child.region_id] == child.to_dict()
    result['model_calls'][0]['region_metadata'] = child.metadata().replace('(0,0,', '(1,0,')
    with pytest.raises(ValueError, match='metadata differs'):
        launch.verify_result_geometry(result, {'p': parent.to_dict()})
    result['model_calls'][0]['region'] = child.to_dict()
    with pytest.raises(ValueError, match='raw acquisition metadata'):
        launch.verify_result_geometry(result, {'p': parent.to_dict()})


def test_partial_format_error_raw_context_overflow_blocks():
    result = invalid()
    for call in result['model_calls']:
        call.update(input_tokens=40900, generation={'max_new_tokens': 2048})
    with pytest.raises(ValueError, match='context contract'):
        launch.audit_raw_call_contracts(result, {}, 40960)


def test_manual_readiness_requires_explicit_missing_trajectory_ack(tmp_path):
    plan = {'root': str(tmp_path), 'validation_readiness_policy': {'manual_acknowledgement_required': True},
            'variants': {name: {'root': str(tmp_path / name)} for name in launch.VARIANTS}}
    paths = []
    for name in launch.VARIANTS:
        path = tmp_path / name / launch.DATASET / 'validation'
        path.mkdir(parents=True)
        paths.append(str(path))
    current_statuses = {'ALL5': {'stage': 'validation_completed_with_model_errors',
        'validation_metrics': {'execution_contract': {'model_protocol_invalid_sample_ids': ['bcnb:3033']}}}}
    # The controller has just verified this case; its previous on-disk status has no invalid set yet.
    atomic_json(tmp_path / 'status.json', {'variants': {}})
    review = {'complete': True, 'plan_hash': digest(plan), 'reviewed_variants': list(launch.VARIANTS),
              'scope': {'images': True, 'model_outputs': True, 'trajectories': True}, 'reviewed_paths': paths,
              'execution_contract_passed': True, 'available_artifacts_reviewed': True}
    path = tmp_path / 'verification/manual_validation_review.json'
    atomic_json(path, review)
    with pytest.raises(ValueError, match='acknowledgement'):
        launch.manual_validation_gate(plan, current_statuses)
    review['model_format_failure_acknowledgement'] = {
        'accepted_for_fixed_rule_evaluation': True, 'available_artifacts_reviewed': True,
        'missing_partial_trajectories_acknowledged': True,
        'invalid_cases': [{'variant': 'ALL5', 'sample_id': 'bcnb:3033'}]}
    atomic_json(path, review)
    assert launch.manual_validation_gate(plan, current_statuses)['passed']
    review['execution_contract_passed'] = False
    atomic_json(path, review)
    with pytest.raises(ValueError, match='rejects'):
        launch.manual_validation_gate(plan, current_statuses)


def failed_scheduler_fixture(tmp_path, monkeypatch, state='FAILED', exit_code='1:0', marker=True, full=False):
    variant, job = 'ALL5', '123'
    directory = tmp_path / 'ALL5'
    runner = directory / 'source/scripts/run_inference.py'
    runner.parent.mkdir(parents=True)
    guard = "raise RuntimeError('Real GPU smoke failed: missing or invalid results')"
    runner.write_text(guard + '\n')
    log_path = directory / 'logs/validation-123_4294967294.log'
    log_path.parent.mkdir()
    ending = 'RuntimeError: Real GPU smoke failed: missing or invalid results' if marker else 'RuntimeError: CUDA out of memory'
    log_path.write_text(f'Traceback (most recent call last):\n  File "{runner}", line 1, in main\n    {guard}\n{ending}\n')
    atomic_json(directory / 'receipts' / (launch.DATASET + '-validation.json'), {
        'job_id': job, 'command': ['sbatch', '--output=' + str(directory / 'logs/validation-%A_%a.log')]})
    plan = {'root': str(tmp_path), 'controller_update': {'kind': 'controller_report_only'},
            'files': {str(runner): sha256_file(runner)},
            'variants': {name: {'root': str(tmp_path / name), 'source': str(tmp_path / name / 'source')}
                         for name in launch.VARIANTS}}
    atomic_json(tmp_path / 'plan.json', plan)
    statuses = {name: {'stage': 'complete' if full else 'validation_passed', 'jobs': {}}
                for name in launch.VARIANTS}
    statuses[variant]['stage'] = 'full_pending' if full else 'validation_pending'
    atomic_json(tmp_path / 'status.json', {'variants': statuses})
    contract = {'passed': True, 'model_protocol_invalid_count': 1,
                'model_protocol_invalid_sample_ids': ['bcnb:3033'], 'verified_successful_trajectories': 27,
                'invalid_without_serialized_partial_trajectory': ['bcnb:3033']}
    gate = {'ready': True, 'passed': False, 'all_answers_valid': False,
            'completion_status': 'validation_completed_with_model_errors', 'profile': {'passed': False},
            'execution_contract': contract, 'metrics': {'expected': 28, 'received': 28,
                'missing': 0, 'invalid': 1, 'execution_contract': contract}}
    monkeypatch.setattr(launch.old, 'queue_rows', lambda: [])
    monkeypatch.setattr(launch, 'submit_once', lambda *args: job)
    monkeypatch.setattr(launch.old, 'job_state', lambda *args: state)
    monkeypatch.setattr(launch.old, 'run_command', lambda args: f'{job}|{state}|{exit_code}\n{job}.batch|{state}|{exit_code}\n')
    monkeypatch.setattr(launch, 'verify_validation', lambda *args: deepcopy(gate))
    monkeypatch.setattr(launch, 'manual_validation_gate', lambda *args: None)
    monkeypatch.setattr(launch, 'generate_report', lambda *args: None)
    def stop_after_iteration(_):
        raise StopIteration('Manual release remains pending in this integration fixture')
    monkeypatch.setattr(launch.time, 'sleep', stop_after_iteration)
    return plan, gate


def test_controller_accepts_only_verified_post_result_smoke_exit_and_preserves_failed_state(tmp_path, monkeypatch):
    plan, gate = failed_scheduler_fixture(tmp_path, monkeypatch)
    with pytest.raises(StopIteration):
        launch.controller(tmp_path)
    state = read_json(tmp_path / 'status.json')
    status = state['variants']['ALL5']
    assert status['stage'] == 'validation_completed_with_model_errors'
    assert status['validation_job_state'] == 'FAILED'
    assert status['validation_exit_code'] == '1:0'
    proof = status['validation_scheduler_failure']
    assert proof['state'] == 'FAILED' and proof['exit_code'] == '1:0'
    assert proof['log_sha256'] == sha256_file(proof['log_path'])
    assert 'smoke guard' in proof['reason']
    stored = read_json(tmp_path / 'verification/validation_ALL5.json')
    assert stored['ready'] and not stored['passed']
    assert stored['scheduler_failure_acceptance'] == proof
    assert state['manual_validation_review_pending'] is True
    assert read_json(tmp_path / 'technical_validation_gate.json')['all_answers_valid'] is False


@pytest.mark.parametrize('state,exit_code,marker', [
    ('FAILED', '2:0', True), ('FAILED', '1:0', False), ('FAILED', '0:9', True),
    ('CANCELLED', '1:0', True), ('TIMEOUT', '1:0', True), ('OUT_OF_MEMORY', '1:0', True)])
def test_controller_other_scheduler_failures_still_block(tmp_path, monkeypatch, state, exit_code, marker):
    failed_scheduler_fixture(tmp_path, monkeypatch, state, exit_code, marker)
    result = launch.controller(tmp_path)
    assert result['variants']['ALL5']['stage'] == 'validation_failed'
    assert not result['complete']
    assert not (tmp_path / 'technical_validation_gate.json').exists()


def test_smoke_exit_does_not_override_incomplete_or_unknown_model_failure(tmp_path, monkeypatch):
    plan, gate = failed_scheduler_fixture(tmp_path, monkeypatch)
    gate['metrics'].update(received=27, missing=1)
    monkeypatch.setattr(launch, 'verify_validation', lambda *args: gate)
    result = launch.controller(tmp_path)
    assert result['variants']['ALL5']['stage'] == 'validation_failed'


def test_smoke_exit_cannot_override_failed_contract_audit(tmp_path, monkeypatch):
    failed_scheduler_fixture(tmp_path, monkeypatch)
    def unknown_failure(*args):
        raise ValueError('Non-format execution/protocol failure: actual context overflow')
    monkeypatch.setattr(launch, 'verify_validation', unknown_failure)
    result = launch.controller(tmp_path)
    assert result['variants']['ALL5']['stage'] == 'validation_failed'
    assert 'context overflow' in result['variants']['ALL5']['error']


def test_full_failed_never_uses_validation_smoke_exit_exception(tmp_path, monkeypatch):
    failed_scheduler_fixture(tmp_path, monkeypatch, full=True)
    monkeypatch.setattr(launch, 'verify_failed_validation_smoke_exit',
                        lambda *args: pytest.fail('Full FAILED must never be overridden'))
    result = launch.controller(tmp_path)
    assert result['variants']['ALL5']['stage'] == 'full_failed'
    assert result['complete'] is False
