"""Paired inference and batch counters for the v2 report."""
from slurm import report_rule_ablations_v2 as report


def rows():
    return [{'sample_id': str(i), 'image_id': str(i // 2), 'choices': {'A': 'one', 'B': 'two'},
             'ground_truth_label': 'A', 'task': 'task'} for i in range(8)]


def predictions(answer):
    return {row['sample_id']: {'status': 'ok', 'pred_answer': answer} for row in rows()}


def test_six_comparison_cluster_intervals_and_incomplete():
    result = report.paired_bootstrap(rows(), predictions('B'), predictions('A'), iterations=256)
    assert result['clusters'] == 4
    assert result['family_size'] == 6
    assert result['delta_accuracy_percent'] == 100
    assert result['delta_ci95_percent'] == [100, 100]
    assert result['delta_family_ci_percent'] == [100, 100]
    assert abs(result['family_confidence_percent'] - 99.16666666666667) < 1e-10
    missing = predictions('A')
    missing.pop('0')
    assert report.paired_bootstrap(rows(), predictions('B'), missing, iterations=256)['status'] == 'withheld_incomplete'


def test_zoom_batch_counts_and_position_metrics():
    row = rows()[0]
    children = [{'region': {'region_id': 'p/z40/' + str(i), 'parent_id': 'p', 'scale': 40},
                 'tissue_status': 'present'} for i in range(5)]
    result = {'status': 'ok', 'pred_answer': 'A', 'process': [
        {'actual_action': 'zoom', 'action_executed': True, 'requested_target_region': 'p',
         'zoom_findings': children, 'decision': {'sufficient': 'No', 'scale_failures': ['nuclei']}},
        {'actual_action': 'conclude', 'findings': children, 'decision': {'sufficient': 'Yes'}}]}
    metrics = report.rule_metrics([row], {row['sample_id']: result})
    assert metrics['counts']['actual_zoom_batches'] == 1
    assert metrics['counts']['zoom_images'] == 5
    assert metrics['counts']['post_zoom_review_contains_children'] == 1
    assert metrics['counts']['evidence_regions'] == 5
    assert metrics['targets']['matching_requests'] == 1
    assert metrics['zoom_scales'] == {'40': 5}
    assert metrics['affected_questions']['actual_zoom'] == ['0']


def test_html_escaping_and_case_names():
    assert report.esc('<image>') == '&lt;image&gt;'
    assert report.case_filename('bcnb:5572') == 'q5572.html'


def test_progress_target_and_generic_count_dedup():
    selected = rows()[:2]
    result = {'status': 'ok', 'pred_answer': 'A',
              'generic_provenance': {'generated_calls_this_process': 3, 'generated_seconds_this_process': 4.5},
              'process': [{'no_progress_count': 2, 'decision': {'sufficient': 'No', 'target_region': 'p',
                 'checks': [{'evidence_type': 'cellular', 'assessable': True, 'coverage_sufficient': False}],
                 'evidence_failures': [{'reason': 'insufficient_field_coverage'}]}}]}
    metrics = report.rule_metrics(selected, {row['sample_id']: result for row in selected})
    assert metrics['generic_generation']['fresh_calls'] == 3
    assert metrics['generic_generation']['fresh_seconds'] == 4.5
    assert metrics['targets']['generated_specified_requests'] == 2
    assert metrics['targets'].get('executed_requests', 0) == 0
    assert metrics['counts']['target_no_progress_events'] == 2
    assert metrics['check_status']['failure_insufficient_field_coverage'] == 2


def test_target_observed_excludes_first_round_and_unexecuted_requests():
    sample = rows()[0]
    result = {'status': 'ok', 'pred_answer': 'A', 'process': [
        {'decision': {'sufficient': 'No', 'target_observed': 'Yes'},
         'actual_action': 'zoom', 'action_executed': True},
        {'decision': {'sufficient': 'No', 'target_observed': 'Yes'},
         'actual_action': None, 'action_executed': False, 'requested_action': 'zoom'},
        {'decision': {'sufficient': 'Yes', 'target_observed': 'Yes'},
         'actual_action': 'conclude', 'action_executed': True}]}
    targets = report.rule_metrics([sample], {sample['sample_id']: result})['targets']
    assert targets['observed_yes'] == 1
    assert targets['post_acquisition_assessments'] == 1
    assert targets['target_observed_without_acquisition'] == 2


def test_protocol_unknown_no_is_separate_from_model_reflection_and_invalid_sources():
    sample = rows()[0]
    result = {'status': 'ok', 'pred_answer': 'A', 'process': [
        {'decision': {'sufficient': 'No', 'reflection_raw': None,
          'reflection_source': 'protocol-format-unknown',
          'evidence_failures': [{'reason': 'invalid_conflict_source', 'region_id': 'invented'},
                               {'reason': 'invalid_primary_answer', 'primary_answer': 'Cannot determine'}]}},
        {'decision': {'sufficient': 'No', 'reflection_raw': {'sufficient': 'Yes'}}},
        {'decision': {'sufficient': 'No', 'reflection_raw': {'sufficient': 'No'}}}]}
    metrics = report.rule_metrics([sample], {sample['sample_id']: result})
    assert metrics['counts']['reflect_format_fallback_events'] == 1
    assert metrics['counts']['raw_reflect_yes'] == 1
    assert metrics['counts']['raw_reflect_no'] == 1
    assert metrics['counts']['raw_yes_effective_no'] == 1
    assert metrics['counts']['invalid_conflict_sources'] == 1
    assert metrics['counts']['primary_refusal_events'] == 1
    assert metrics['affected_questions']['reflect_format_fallback'] == [sample['sample_id']]
    assert metrics['affected_questions']['invalid_conflict_source'] == [sample['sample_id']]


def test_native_artifact_pixels_are_verified(tmp_path):
    import hashlib
    from PIL import Image
    import pytest
    path = tmp_path / 'input.png'
    image = Image.new('RGB', (2, 2), 'white')
    image.save(path)
    call = {'image_artifact_path': str(path), 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'input_size': [2, 2]}
    assert report.verified_native_image(call) == str(path)
    call['pixel_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='recorded pixels'):
        report.verified_native_image(call)


def test_generic_only_evidence_image_and_separated_round_text(tmp_path):
    import hashlib
    from PIL import Image
    path = tmp_path / 'generic.png'
    image = Image.new('RGB', (2, 2), 'pink')
    image.save(path)
    artifact = {'image_artifact_path': str(path), 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'input_size': [2, 2], 'input_origin': 'cached_generic_crop'}
    finding = {'region': {'region_id': 'generic-parent', 'scale': 5}, 'image_artifact': artifact,
               'generic_description': '<generic observation>', 'question_description': ''}
    result = {'process': [{'iteration': 1, 'findings': [finding]}, {'iteration': 2, 'findings': [finding]}]}
    cards = report.image_cards(result)
    assert cards.count('<img ') == 1
    assert '缓存 Generic 对应区域 RGB；未新增描述调用。' in cards
    rounds = report.evidence_rounds(result)
    assert '各轮证据：Generic / Question' in rounds
    assert '&lt;generic observation&gt;' in rounds
    assert '未生成' in rounds


def test_full_report_contains_real_transitions_and_collapsed_case_outputs(tmp_path):
    from data_processing.common import atomic_json, digest, write_manifest, read_manifest
    samples = rows()
    for sample in samples:
        sample.update(dataset='bcnb', question='Which <feature>?', image_path='/unused.jpg')
    manifest_path = tmp_path / 'full.json'
    write_manifest(manifest_path, samples, dataset='bcnb', images_verified=True)
    manifest = read_manifest(manifest_path)
    plan = {'baseline_root': str(tmp_path / 'baseline'),
            'datasets': {report.DATASET: {'full_manifest': str(manifest_path), 'baseline_expected_correct': 8}},
            'variants': {}, 'comparison_limitation': 'Common flow changed.',
            'source_version_note': 'Three old arms and three new arms retain their actual hashes.'}
    for name in ('BASELINE',) + report.FAMILY:
        arm = tmp_path / ('baseline' if name == 'BASELINE' else name)
        directory = arm / report.DATASET
        write_manifest(directory / 'manifests/shard_0.json', samples, dataset='bcnb', images_verified=True)
        config, prompt = {'protocol': 'fixture'}, {'templates': {}}
        source_hash = 'fixture-source-' + name
        identity = {'config': config, 'code_hash': source_hash, 'samples_hash': manifest['samples_hash'],
                    'protocol': 'fixture', 'prompt_manifest_hash': digest(prompt)}
        run_hash = digest(identity)
        shard = directory / 'full/shard_0'
        atomic_json(shard / 'run_lock.json', {**identity, 'run_hash': run_hash})
        atomic_json(shard / 'prompt_manifest.json', prompt)
        for sample in samples:
            result = {'sample_id': sample['sample_id'], 'image_id': sample['image_id'],
                      'run_hash': run_hash, 'status': 'ok', 'pred_answer': 'A' if name == 'BASELINE' else 'B',
                      'process': [{'decision': {'answer': 'A', 'sufficient': 'Yes'}, 'action': 'conclude'}],
                      'model_calls': [{'stage': 'final', 'raw_response': '<unsafe output>'}]}
            atomic_json(shard / 'results' / (digest(sample['sample_id']) + '.json'), result)
        if name != 'BASELINE':
            plan['variants'][name] = {'root': str(arm), 'source_hash': source_hash,
                'source_version': {'campaign_root': str(tmp_path / ('new' if name in ('R1_EVIDENCE', 'R2_DIFFERENTIAL', 'ALL5') else 'old')),
                    'validation_source_hash': source_hash, 'full_source_hash': source_hash},
                'prompt_manifest_hash': digest(prompt), 'datasets': {report.DATASET: {'config_hash': digest(config)}}}
    atomic_json(tmp_path / 'plan.json', plan)
    values = report.build_report(tmp_path, iterations=32)
    report.write_report(tmp_path, *values)
    output = report.old.read_json(tmp_path / 'report.json')
    assert output['complete']
    assert output['arms']['R1_EVIDENCE']['paired_with_baseline']['overall']['transitions']['correct_wrong'] == 8
    assert output['case_pages'] == 8
    assert output['manual_review']['status'] == 'pending'
    assert '六比较校正区间' in (tmp_path / 'report.zh.html').read_text()
    assert '跨组 source hash 不同' in (tmp_path / 'report.zh.html').read_text()
    assert output['source_versions']['ALL5']['full_source_hash'] == 'fixture-source-ALL5'
    detail = (tmp_path / 'cases/q0.html').read_text()
    assert '&lt;unsafe output&gt;' in detail
    assert detail.count('<summary>') >= 14
