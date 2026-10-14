"""CPU checks for rules-v2 execution constraints; no model inference."""
import json
import hashlib
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from data_processing.regions import Region
from models.inference import ModelProtocolError
from models.rules_v2_inference import RulesV2ModelBackend, normalize_tissue_response, parse_tissue_response
from models.rules_v2_prompts import flags_for_variant, templates_for_config


SAMPLE = {'question': 'Which grade?', 'choices': {'A': '1', 'B': '2', 'C': '3'}}


def finding(name='parent', scale=5, parent=None, status=None):
    value = {'region': Region(name, '/unused', 0, 0, 4096, 4096, 1024, 1024,
                              'physical', scale, .5, 'assumed', parent).to_dict(),
             'description': 'Claimed observation at 40x, without trusted scale.'}
    if status:
        value['tissue_status'] = status
    return value


def predict(kind='fine_detail', feature='mitotic figures'):
    return {'answer': 'C', 'thinking_steps': feature + ' supports this estimate',
            'required_features': [{'feature': feature, 'evidence_type': kind}]}


def check(kind='fine_detail', names=None, feature='mitotic figures', **kwargs):
    return {'feature': feature, 'evidence_type': kind,
            'region_ids': ['parent'] if names is None else names,
            'assessable': True, 'coverage_sufficient': True, **kwargs}


def missing(zoom='No', level=None, **kwargs):
    return {'missing_info': 'mitotic figures', 'zoom_recommendation': zoom,
            'recommended_zoom_level': level, 'zoom_reason': 'inspect local detail', **kwargs}


class StubBackend(RulesV2ModelBackend):
    def __init__(self, variant, responses):
        self.config = {'rules_v2': {'variant': variant},
                       'algorithm': {'summary_threshold': 20, 'summary_chunk_size': 10}}
        self.rule_flags = flags_for_variant(variant)
        self.rule_templates = templates_for_config(self.config)
        self.tissue_status = {}
        self.responses = list(responses)
        self.records = []
        self.generation = {'step_a_tokens': 1024, 'step_b_tokens': 1024,
                           'step_c_tokens': 1024, 'final_tokens': 1024,
                           'summary_tokens': 1024, 'retries': 1}
        self.context_limit = 100000

    def _evidence_for_v2(self, findings, system, render, tokens, question, context, **kwargs):
        return 'Actual metadata with separated narrative'

    def _text(self, messages, tokens, stage, greedy=False):
        self.records.append({'stage': stage, 'messages': deepcopy(messages)})
        return json.dumps(self.responses.pop(0))


def run(backend, findings=None, context=None):
    findings = findings if findings is not None else [finding()]
    context = context or {'eligible_zoom_parents': [finding()['region']],
                          'current_zoom_parents': [finding()['region']]}
    return backend.evaluate(findings, SAMPLE, 'physical', [10, 20, 40], context)


def test_scale_uses_trusted_metadata_and_enforces_zoom_floor():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes', 'checks': [check()]},
                                          missing()])
    result = run(backend)
    assert result['reflection_raw']['sufficient'] == 'Yes'
    assert result['sufficient'] == result['effective_sufficient'] == 'No'
    assert result['zoom_recommendation'] == 'Yes'
    assert result['zoom_level'] == 40
    assert result['missing_raw']['zoom_recommendation'] == 'No'
    assert result['scale_failures'][0]['required_scale'] == 40


@pytest.mark.parametrize('kind,floor', [('overview', 5), ('architecture', 10),
                                      ('cellular', 20), ('fine_detail', 40)])
def test_each_required_type_accepts_actual_floor(kind, floor):
    feature = {'overview': 'distribution', 'architecture': 'glands',
               'cellular': 'nuclear pleomorphism', 'fine_detail': 'mitotic figures'}[kind]
    backend = StubBackend('R1_EVIDENCE', [predict(kind, feature), {'sufficient': 'Yes', 'checks': [check(kind, feature=feature)]}])
    result = run(backend, [finding(scale=floor)])
    assert result['sufficient'] == 'Yes'
    assert not result['scale_failures']


def test_low_context_and_high_detail_can_be_cited_together():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes',
                                                     'checks': [check(names=['parent', 'detail'])]}])
    result = run(backend, [finding(), finding('detail', 40, 'parent')])
    assert result['sufficient'] == 'Yes'


def test_scalar_reflect_wire_preserves_raw_and_maps_to_internal_region_ids():
    wire = check()
    wire.pop('region_ids')
    wire['region_id'] = 'parent'
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes', 'checks': [wire]}, missing()])
    result = run(backend)
    assert result['reflection_raw']['checks'][0]['region_id'] == 'parent'
    assert 'region_ids' not in result['reflection_raw']['checks'][0]
    assert result['checks'][0]['region_ids'] == ['parent']
    assert result['sufficient'] == 'No' and result['zoom_level'] == 40
    assert wire['region_id'] == 'parent' and 'region_ids' not in wire


def test_scalar_and_conflicting_legacy_anchor_is_not_silently_normalized():
    backend = StubBackend('R1_EVIDENCE', [])
    assert not backend._valid_reflect({'sufficient': 'Yes',
        'checks': [check(region_id='parent', names=['different'])]})


def test_scalar_actual_high_scale_passes_with_global_coverage_assessed():
    wire = check()
    wire.pop('region_ids')
    wire['region_id'] = 'detail'
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes', 'checks': [wire]}])
    result = run(backend, [finding(), finding('detail', 40, 'parent')])
    assert result['sufficient'] == 'Yes'
    assert result['checks'][0]['coverage_sufficient'] is True


@pytest.mark.parametrize('feature', ['mitosis', 'mitoses', 'mitotic activity', 'mitotic_count',
                                    'mitotic_counts', 'mitotic_figures', 'mitotic figure'])
def test_explicit_mitotic_features_cannot_downgrade_fixed_scale_policy(feature):
    backend = StubBackend('R1_EVIDENCE', [predict('cellular', feature),
        {'sufficient': 'Yes', 'checks': [check('cellular', ['detail20'], feature)]}, missing()])
    result = run(backend, [finding(), finding('detail20', 20, 'parent')])
    assert result['prediction_raw']['required_features'][0]['evidence_type'] == 'cellular'
    assert result['required_features'][0]['feature'] == feature
    assert result['required_features'][0]['evidence_type'] == 'fine_detail'
    assert result['type_policy_adjustments'][0]['required_scale'] == 40
    assert result['sufficient'] == 'No' and result['zoom_level'] == 40
    assert result['scale_failures'][0]['required_scale'] == 40
    reflect_message = next(r for r in backend.records if r['stage'] == 'reflect')['messages'][1]['content']
    assert '"evidence_type": "fine_detail"' in reflect_message


def test_duplicate_low_scale_checks_still_record_and_enforce_floor():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes',
        'checks': [check(), check()]}, missing()])
    result = run(backend)
    assert result['sufficient'] == 'No'
    assert any(f['reason'] == 'missing_or_duplicate_check' for f in result['evidence_failures'])
    assert result['scale_failures'][0]['region_ids'] == ['parent']
    assert result['zoom_level'] == 40


def test_duplicate_checks_with_already_adequate_scale_do_not_force_more_zoom():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes',
        'checks': [check(names=['parent']), check(names=['detail40'])]}, missing()])
    result = run(backend, [finding(), finding('detail40', 40, 'parent')])
    assert result['sufficient'] == 'No'
    assert result['scale_failures'] == []
    assert result['zoom_recommendation'] == 'No'
    assert 'action_override' not in result


def test_duplicate_historical_low_checks_do_not_force_unrelated_current_zoom():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes',
        'checks': [check(), check()]}, missing()])
    old, current = finding(), finding('current')
    result = run(backend, [old, current], {'eligible_zoom_parents': [old['region'], current['region']],
                                        'current_zoom_parents': [current['region']]})
    assert result['sufficient'] == 'No' and result['scale_failures']
    assert result['zoom_recommendation'] == 'No'
    assert result['scale_target_unavailable']


def test_mitotic_type_policy_is_isolated_from_other_arms():
    original = predict('cellular', 'mitotic_activity')
    for variant in ('R2_DIFFERENTIAL', 'R3_ROI_VALIDITY', 'R4_VISUAL_MISSING', 'R5_TARGET_REGION'):
        backend = StubBackend(variant, [])
        normalized, changes = backend._apply_type_policy(original)
        assert normalized == original and changes == []


def test_r2_unlisted_alternative_is_insufficient_without_inventing_or_retrying_it():
    invalid = {'answer': 'A', 'thinking_steps': 'estimated from morphology',
               'alternative_answer': 'Cannot determine from H&E alone',
               'discriminating_feature': 'nuclei', 'conflicts': []}
    backend = StubBackend('R2_DIFFERENTIAL', [invalid,
        {'sufficient': 'Yes', 'unresolved_conflicts': []}, missing()])
    result = run(backend)
    assert result['alternative_answer'] == invalid['alternative_answer']
    assert result['sufficient'] == 'No'
    assert result['evidence_failures'][0]['reason'] == 'invalid_differential'
    assert len([r for r in backend.records if r['stage'] == 'predict']) == 1


@pytest.mark.parametrize('alternative', ['Equivocal', 'Negative', 'B'])
def test_real_binary_equivocal_or_same_label_is_recorded_as_differential_failure(alternative):
    sample = {'question': 'Determine HER2 status', 'choices': {'A': 'Positive', 'B': 'Negative'}}
    prediction = {'answer': 'Negative', 'thinking_steps': 'morphological estimate',
                  'alternative_answer': alternative, 'discriminating_feature': 'nuclei', 'conflicts': []}
    backend = StubBackend('R2_DIFFERENTIAL', [prediction, {'sufficient': 'Yes', 'unresolved_conflicts': []}, missing()])
    result = backend.evaluate([finding()], sample, 'physical', [10, 20, 40])
    assert result['answer'] == 'Negative' and result['alternative_answer'] == alternative
    assert result['sufficient'] == 'No'
    assert result['evidence_failures'][0]['reason'] == 'invalid_differential'


def test_r2_invalid_reporting_or_conflict_structure_still_uses_specific_retry():
    invalid = {'answer': 'Cannot determine from H&E alone', 'thinking_steps': 'estimate',
               'alternative_answer': 'B', 'discriminating_feature': 'nuclei', 'conflicts': []}
    valid = {**invalid, 'answer': 'A'}
    invalid['thinking_steps'] = None
    backend = StubBackend('R2_DIFFERENTIAL', [invalid, valid,
        {'sufficient': 'Yes', 'unresolved_conflicts': []}])
    result = run(backend)
    assert result['answer'] == 'A'
    retry = backend.records[1]['messages'][-1]['content']
    assert 'two distinct listed choices' in retry
    assert 'Cannot determine from H&E alone' in retry
    assert '"A": "1"' in retry
    invalid['answer'] = 'A'
    invalid['conflicts'] = [{'region_id': 'parent', 'feature': 'nuclei', 'claims': ['only one claim']}]
    assert not backend._valid_predict(invalid, backend._registry([finding()]), SAMPLE['choices'])


def test_final_retry_requires_listed_best_estimate_and_no_default_negative():
    backend = StubBackend('R2_DIFFERENTIAL', [
        {'answer': 'Cannot determine from H&E alone', 'explanation': 'uncertain'},
        {'answer': 'B', 'explanation': 'indirect estimate with uncertainty'},
    ])
    result = backend.conclude([finding()], [], SAMPLE)
    assert result['answer'] == 'B'
    retry = backend.records[-1]['messages'][-1]['content']
    assert 'Return one listed answer choice or label' in retry
    assert 'Missing assay data is not a Negative result' in retry
    assert 'Cannot determine from H&E alone' in retry


@pytest.mark.parametrize('checks,reason', [([], 'missing_or_duplicate_check'),
                                         ([check(names=['invented'])], 'missing_or_invalid_region'),
                                         ([check(names=[])], 'missing_or_invalid_region'),
                                         ([check(assessable=False)], 'unassessable'),
                                         ([check(coverage_sufficient=False)], 'insufficient_field_coverage')])
def test_missing_invalid_unassessable_or_uncovered_required_evidence_rejects(checks, reason):
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes', 'checks': checks}, missing()])
    result = run(backend, [finding(scale=40)])
    assert result['sufficient'] == 'No'
    assert reason in {item['reason'] for item in result['evidence_failures']}


def test_scale_shortfall_without_relevant_eligible_parent_does_not_force_zoom():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes', 'checks': [check()]}, missing()])
    result = run(backend, context={'eligible_zoom_parents': []})
    assert result['sufficient'] == 'No'
    assert result['zoom_recommendation'] == 'No'


def test_historical_requirement_cannot_force_zoom_in_unrelated_current_parent():
    backend = StubBackend('R1_EVIDENCE', [predict(), {'sufficient': 'Yes', 'checks': [check()]}, missing()])
    old, current = finding(), finding('current')
    context = {'eligible_zoom_parents': [old['region'], current['region']],
               'current_zoom_parents': [current['region']]}
    result = run(backend, [old, current], context)
    assert result['sufficient'] == 'No'
    assert result['scale_failures'][0]['region_ids'] == ['parent']
    assert result['zoom_recommendation'] == 'No'
    assert 'action_override' not in result
    assert result['scale_target_unavailable'][0]['reason'] == 'NeedsHigherScaleButNoCurrentTarget'
    assert result['scale_target_unavailable'][0]['cited_parent_regions'] == ['parent']
    assert result['scale_target_unavailable'][0]['execution_parent_regions'] == ['current']


@pytest.mark.parametrize('target,forced', [(None, False), ('parent', True)])
def test_all_historical_scale_override_requires_explicit_historical_target(target, forced):
    prediction = {**predict(), 'alternative_answer': 'B',
                  'discriminating_feature': 'mitoses', 'conflicts': []}
    reflection = {'sufficient': 'Yes', 'checks': [check()],
                  'unresolved_conflicts': [], 'target_observed': 'Unknown'}
    request = missing('Yes' if target else 'No', 10 if target else None,
                      evidence_type='fine_detail', target_region=target)
    backend = StubBackend('ALL5', [prediction, reflection, request])
    old, current = finding(), finding('current')
    context = {'eligible_zoom_parents': [old['region'], current['region']],
               'current_zoom_parents': [current['region']]}
    result = run(backend, [old, current], context)
    assert result['sufficient'] == 'No'
    assert result['target_region'] == target
    assert ('action_override' in result) is forced
    assert result['zoom_level'] == (40 if forced else None)


def test_same_region_unresolved_conflict_rejects_raw_yes():
    prediction = {'answer': 'C', 'thinking_steps': 'invasion', 'alternative_answer': 'B',
                  'discriminating_feature': 'stromal invasion',
                  'conflicts': [{'region_id': 'parent', 'feature': 'invasion',
                                 'claims': ['present', 'absent']}]}
    backend = StubBackend('R2_DIFFERENTIAL', [prediction, {'sufficient': 'Yes',
        'unresolved_conflicts': [{'region_id': 'parent', 'feature': 'invasion'}]}, missing()])
    result = run(backend)
    assert result['sufficient'] == 'No'
    assert result['evidence_failures'][0]['reason'] == 'unresolved_conflict'


def test_all_absent_tissue_forces_explore_without_assuming_negative():
    backend = StubBackend('R3_ROI_VALIDITY', [ {'answer': 'C', 'thinking_steps': 'uncertain'},
                                             {'sufficient': 'Yes'}, missing('Yes', 40)])
    result = run(backend, [finding(status='absent')])
    assert result['answer'] == 'C'
    assert result['sufficient'] == 'No'
    assert result['zoom_recommendation'] == 'No'
    assert result['zoom_level'] is None


@pytest.mark.parametrize('status', ['present', 'uncertain', None])
def test_unknown_or_sparse_tissue_is_not_automatically_rejected(status):
    backend = StubBackend('R3_ROI_VALIDITY', [{'answer': 'B', 'thinking_steps': 'evidence'}, {'sufficient': 'Yes'}])
    result = run(backend, [finding(status=status)])
    assert result['sufficient'] == 'Yes'


def test_invalid_target_gets_format_retry_and_valid_target_is_preserved():
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'detail'},
        {'sufficient': 'No'}, missing('Yes', 20, target_region='invented'),
        missing('Yes', 20, target_region='parent')])
    result = run(backend)
    assert result['target_region'] == 'parent'
    assert len([record for record in backend.records if record['stage'] == 'missing_info']) == 2
    assert len(backend.records[-1]['messages']) == 3


def test_persistently_invalid_target_never_silently_retargets():
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'detail'},
        {'sufficient': 'No'}, missing('Yes', 20, target_region='invented'),
        missing('Yes', 20, target_region='another_invented')])
    with pytest.raises(ModelProtocolError, match='Invalid missing_info'):
        run(backend)


def test_explore_unused_target_is_normalized_and_recorded():
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'detail'},
        {'sufficient': 'No'}, missing(target_region='parent')])
    result = run(backend)
    assert result['target_region'] is None
    assert result['unused_target'] == 'parent'
    assert result['missing_raw']['target_region'] == 'parent'
    assert len(backend.records[-1]['messages']) == 2


def test_quoted_null_is_harmless_target_value_normalization_with_raw_preserved():
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'detail'},
        {'sufficient': 'No'}, missing(target_region='null')])
    result = run(backend)
    assert result['target_region'] is None
    assert result['missing_raw']['target_region'] == 'null'
    assert result['missing_value_normalization']['target_region']['from'] == 'null'
    assert result['missing_value_normalization']['target_region']['to'] is None
    assert len([r for r in backend.records if r['stage'] == 'missing_info']) == 1


def test_nonempty_target_for_explore_is_unused_and_not_retargeted():
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'detail'},
        {'sufficient': 'No'}, missing(target_region='parent')])
    result = run(backend)
    assert result['target_region'] is None
    assert result['missing_raw']['target_region'] == 'parent'
    assert result['unused_target'] == 'parent'
    calls = [r for r in backend.records if r['stage'] == 'missing_info']
    assert len(calls) == 1
    assert result['zoom_recommendation'] == 'No'


def test_real_pilot6670_no_zoom_omitted_level_is_conditionally_optional():
    request = {"missing_info": "The molecular subtype cannot be definitively determined from the image alone. While the image shows features consistent with a triple-negative breast cancer, such as infiltrative growth, nuclear atypia, and frequent mitotic figures, the absence of ER/PR expression and HER2 status would require immunohistochemical testing to confirm. The image does not show the molecular testing results explicitly.",
               "zoom_recommendation": "No", "zoom_reason": "The image does not show molecular testing results, and zooming in would not provide additional information about the molecular subtype.", "target_region": None}
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'estimate'},
                                              {'sufficient': 'No'}, request])
    result = run(backend)
    assert result['zoom_level'] is None and result['target_region'] is None
    assert result['missing_info'] == request['missing_info']
    assert result['zoom_reason'] == request['zoom_reason']
    assert 'recommended_zoom_level' not in result['missing_raw']
    assert result['missing_value_normalization']['recommended_zoom_level']['from'] == '<omitted>'
    assert len([r for r in backend.records if r['stage'] == 'missing_info']) == 1


@pytest.mark.parametrize('zoom,level', [('No', None), ('Yes', 20)])
def test_optional_r5_target_omission_is_null_without_retarget(zoom, level):
    request = missing(zoom, level)
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'estimate'},
                                              {'sufficient': 'No'}, request])
    result = run(backend)
    assert result['target_region'] is None
    assert 'target_region' not in result['missing_raw']
    assert result['zoom_recommendation'] == zoom


def test_yes_illegal_child_target_still_retries_instead_of_ignoring():
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'estimate'},
        {'sufficient': 'No'}, missing('Yes', 40, target_region='parent/z40.0/0_0'),
        missing('Yes', 40, target_region='parent')])
    result = run(backend)
    assert result['target_region'] == 'parent'
    assert 'unused_target' not in result
    assert len([r for r in backend.records if r['stage'] == 'missing_info']) == 2


def test_yes_missing_zoom_level_remains_a_required_field():
    request = missing('Yes', 20)
    del request['recommended_zoom_level']
    backend = StubBackend('R5_TARGET_REGION', [{'answer': 'B', 'thinking_steps': 'estimate'},
        {'sufficient': 'No'}, request, missing('Yes', 20, target_region=None)])
    result = run(backend)
    assert result['zoom_level'] == 20
    assert len([r for r in backend.records if r['stage'] == 'missing_info']) == 2


def test_r4_receives_executed_request_and_actual_progress():
    backend = StubBackend('R4_VISUAL_MISSING', [{'answer': 'B', 'thinking_steps': 'uncertain'},
        {'sufficient': 'No', 'target_observed': 'No'}, missing(evidence_type='cellular')])
    context = {'last_request': {'missing_info': 'nuclear pleomorphism'},
               'last_new_findings': [{'region': 'actual_child', 'description': 'not assessed'}],
               'no_progress_count': 2, 'eligible_zoom_parents': []}
    result = run(backend, context=context)
    assert result['target_observed'] == 'No'
    for record in backend.records:
        user = record['messages'][1]['content']
        assert 'nuclear pleomorphism' in user
        assert 'actual_child' in user
        expected = 3 if record['stage'] == 'missing_info' else 2
        assert '"no_progress_count": ' + str(expected) in user


def test_second_failed_acquisition_is_visible_to_immediately_following_missing():
    backend = StubBackend('R4_VISUAL_MISSING', [{'answer': 'B', 'thinking_steps': 'uncertain'},
        {'sufficient': 'No', 'target_observed': 'No'}, missing(evidence_type='cellular')])
    context = {'last_request': {'missing_info': 'nuclear pleomorphism', 'actual_action': 'zoom'},
               'last_new_findings': [finding('detail', 20, 'parent')],
               'no_progress_count': 1, 'eligible_zoom_parents': []}
    result = run(backend, context=context)
    assert result['current_no_progress_count'] == 2
    assert '"no_progress_count": 2' in backend.records[-1]['messages'][1]['content']
    assert '"no_progress_count": 1' in backend.records[-2]['messages'][1]['content']


def test_perceptor_generic_and_question_use_effective_template():
    backend = StubBackend('R2_DIFFERENTIAL', [])
    prompt = backend.perceptor_user('Region=parent;', 'Which subtype?', 'nuclear atypia')
    assert 'Input modality: H&E' in prompt
    assert 'Separate assessed features' in prompt
    assert 'nuclear atypia' in prompt
    assert 'visible H&E morphology' in backend.rule_templates['PERCEPTOR_SYSTEM']


def test_r3_generic_remains_original_and_only_question_requests_json():
    backend = StubBackend('R3_ROI_VALIDITY', [])
    assert 'tissue_status' not in backend.perceptor_user('Region=parent;')
    assert 'tissue_status' in backend.perceptor_user('Region=parent;', 'Which?')
    assert 'JSON object inside <answer>' not in backend.perceptor_system()
    assert 'JSON object inside <answer>' in backend.perceptor_system('Which?')


def test_r3_known_key_case_and_space_variation_is_harmless():
    raw = {'Tissue Status': 'present', 'Findings': 'visible tissue'}
    parsed, renamed = normalize_tissue_response(raw)
    assert parsed == {'tissue_status': 'present', 'findings': 'visible tissue'}
    assert renamed == {'Tissue Status': 'tissue_status', 'Findings': 'findings'}
    assert raw == {'Tissue Status': 'present', 'Findings': 'visible tissue'}


def test_explicit_tags_allow_unescaped_quotes_in_findings():
    raw = '<answer><tissue_status>uncertain</tissue_status><findings>The word "uncertain" is quoted; sparse cells.</findings></answer>'
    parsed, renamed, wire = parse_tissue_response(raw)
    assert parsed == {'tissue_status': 'uncertain', 'findings': 'The word "uncertain" is quoted; sparse cells.'}
    assert renamed == {} and wire == 'tags'


def test_valid_legacy_json_is_compatible_with_tag_protocol():
    parsed, renamed, wire = parse_tissue_response('{"Tissue Status":"present","Findings":"visible tissue"}')
    assert parsed == {'tissue_status': 'present', 'findings': 'visible tissue'}
    assert renamed and wire == 'json'


@pytest.mark.parametrize('raw', [
    '<tissue_status>present</tissue_status>',
    '<findings>visible cells</findings>',
    '<tissue_status>Unknown</tissue_status><findings>visible cells</findings>',
    '<tissue_status>absent</tissue_status><findings> </findings>',
    '<tissue_status>present</tissue_status><tissue_status>absent</tissue_status><findings>cells</findings>',
    '<tissue_status>present</tissue_status><findings>one</findings><findings>two</findings>',
    '{"tissue_status":"uncertain","findings":"bad quote {"tissue_status":"absent","findings":"nested"}}',
])
def test_invalid_tags_or_malformed_json_never_infer_absent(raw):
    parsed, _, _ = parse_tissue_response(raw)
    assert parsed is None


def test_real_findings_uncertain_quotes_are_repaired_with_provenance():
    raw = '<answer>{"tissue_status":"uncertain","findings":"The tissue is "uncertain" due to sparse cells."}</answer>'
    parsed, renamed, wire, repair = parse_tissue_response(raw, return_repair=True)
    assert parsed == {'tissue_status': 'uncertain', 'findings': 'The tissue is "uncertain" due to sparse cells.'}
    assert wire == 'json' and renamed == {}
    assert repair['inserted_backslashes'] == 2
    assert repair['raw_container'] == raw[len('<answer>'):-len('</answer>')]
    assert json.loads(repair['repaired_json']) == parsed


def test_archived_all5379_actual_quote_failure_is_repaired_without_changing_findings():
    raw = '''```json
{
  "tissue_status": "present",
  "findings": "The image shows a breast tumor with moderate nuclear atypia and scattered mitotic figures. However, the absence of high-power magnification limits definitive assessment of nuclear pleomorphism and mitotic count. The background contains necrotic debris, which is nonspecific but raises suspicion for malignancy. These features are insufficient to confirm the histological grade, necessitating immunohistochemical analysis (e.g., ER/PR/HER2) for further classification. The findings are categorized as "uncertain" due to insufficient morphological detail at the observed scale."
}
```'''
    parsed, _, _, repair = parse_tissue_response(raw, return_repair=True)
    assert parsed['tissue_status'] == 'present'
    assert parsed['findings'] == raw.split('"findings": "', 1)[1].rsplit('"\n}', 1)[0]
    assert 'necessitating immunohistochemical analysis' in parsed['findings']
    assert repair['inserted_backslashes'] == 2


def test_perceptor_syntax_repair_is_recorded_and_preserves_raw_without_retrieval_label_changes(monkeypatch):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    raw = '<answer>{"tissue_status":"present","findings":"The feature is "uncertain"."}</answer>'
    _mock_perceptor(backend, monkeypatch, [raw])
    result = backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert result == 'The feature is "uncertain".'
    assert backend.tissue_status == {'parent': 'present'}
    assert len(backend.records) == 1
    assert backend.records[0]['raw_response'] == raw
    assert backend.records[0]['syntax_repair']['inserted_backslashes'] == 2


@pytest.mark.parametrize('raw', [
    '{"tissue_status":"uncertain"}',
    '{"findings":"The tissue is "uncertain"."}',
    '{"tissue_status":"unknown","findings":"The tissue is "uncertain"."}',
    '{"tissue_status":"present","tissue_status":"absent","findings":"The tissue is "uncertain"."}',
    '{"tissue_status":"present","Tissue Status":"present","findings":"cells"}',
    '{"tissue_status":"uncertain","findings":"The tissue is "uncertain".","extra":"must not swallow"}',
    '{"tissue_status":"uncertain","findings":"The tissue is "uncertain".","findings":"duplicate"}',
    '{"tissue_status":"uncertain","findings":"The tissue is "uncertain"."',
])
def test_quote_repair_rejects_missing_duplicate_extra_non_enum_and_incomplete_objects(raw):
    parsed, _, _, repair = parse_tissue_response(raw, return_repair=True)
    assert parsed is None and repair is None


def test_real_tag_retry_parses_explicit_enum_and_free_text(monkeypatch):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    _mock_perceptor(backend, monkeypatch, [
        '{"tissue_status":"uncertain"}',
        '<tissue_status>uncertain</tissue_status><findings>The word "uncertain" is quoted.</findings>',
    ])
    result = backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert result == 'The word "uncertain" is quoted.'
    assert backend.tissue_status == {'parent': 'uncertain'}
    assert backend.records[-1]['question_wire_format'] == 'tags'
    assert '"tissue_status": "[one allowed value]"' in backend.records[-1]['messages'][-1]['content']


@pytest.mark.parametrize('raw', [
    {'Tissue Status': 'long narrative instead of an enum'},
    {'tissue_status': 'present'},
    {'findings': 'visible tissue'},
    {'tissue_status': None, 'findings': 'visible tissue'},
    {'tissue_status': [], 'findings': 'visible tissue'},
    {'tissue_status': 'Unknown', 'findings': 'visible tissue'},
    {'tissue_status': 'absent', 'findings': ''},
    {'tissue_status': 'present', 'Tissue Status': 'absent', 'findings': 'ambiguous'},
    {'tissue_status': 'present', 'findings': 'one', 'Findings': 'different'},
])
def test_r3_incomplete_or_ambiguous_response_is_never_repaired_to_absent(raw):
    parsed, _ = normalize_tissue_response(raw)
    assert parsed is None


def _mock_perceptor(backend, monkeypatch, outputs):
    import torch

    class Inputs(dict):
        def __getattr__(self, name):
            return self[name]

        def to(self, device):
            return self

    class Processor:
        image_processor = SimpleNamespace(merge_size=2)

        def apply_chat_template(self, messages, **kwargs):
            return 'formatted vision request'

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 2, 3]]),
                          image_grid_thw=torch.tensor([[1, 2, 2]]))

        def decode(self, tokens, **kwargs):
            return outputs.pop(0)

    backend.processor = Processor()
    backend.perceptor = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=9, to_dict=lambda: {}),
        generate=lambda **kwargs: torch.tensor([[1, 2, 3, 9]]))
    backend.generation.update(question_description_tokens=2048, generic_description_tokens=1024)
    backend._seed = lambda stage: 128
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils', SimpleNamespace(
        process_vision_info=lambda messages: ([messages[1]['content'][0]['image']], None)))


def test_real_perceptor_format_retry_uses_exact_schema_and_preserves_known_key_alias(monkeypatch):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    _mock_perceptor(backend, monkeypatch, [
        '{"Tissue Status":"long prose without enumerated status"}',
        '{"Tissue Status":"present","Findings":"visible tissue"}',
    ])
    description = backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert description == 'visible tissue'
    assert backend.tissue_status == {'parent': 'present'}
    assert len(backend.records) == 2
    final = backend.records[-1]
    assert final['parsed_response'] == {'tissue_status': 'present', 'findings': 'visible tissue'}
    assert final['key_normalization'] == {'Tissue Status': 'tissue_status', 'Findings': 'findings'}
    assert final['raw_response'] == '{"Tissue Status":"present","Findings":"visible tissue"}'
    retry = final['messages'][-1]['content']
    assert '"tissue_status"' in retry and '"findings"' in retry
    assert 'long prose without enumerated status' in retry
    assert 'JSON object inside <answer>' in final['messages'][0]['content']


def test_perceptor_two_invalid_responses_raise_without_absent_default(monkeypatch):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    _mock_perceptor(backend, monkeypatch, [
        '{"Tissue Status":"unknown narrative"}',
        '{"Tissue Status":"present"}',
    ])
    with pytest.raises(ModelProtocolError, match='after two attempts'):
        backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert backend.tissue_status == {}
    assert len(backend.records) == 2


@pytest.mark.parametrize('observation', [
    'The tissue is present and classified as a low-grade breast carcinoma.',
    '**Tissue status**: Present (breast tumor).\n**Findings**: Nuclear atypia is observed.',
    'Only background is visible; no reliable assessment can be made.',
])
def test_completed_prose_without_schema_retains_unknown_status_not_prose_inference(monkeypatch, observation):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    raw = '<answer>' + observation + '</answer>'
    _mock_perceptor(backend, monkeypatch, [raw, raw])
    returned = backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert returned == observation
    assert backend.tissue_status == {'parent': 'uncertain'}
    assert backend.tissue_status_source == {'parent': 'protocol-format-unknown'}
    assert len(backend.records) == 2
    assert all(call['raw_response'] == raw for call in backend.records)
    assert backend.records[-1]['format_fallback']['status_inferred'] is False
    registry = backend._registry([dict(finding(), question_description=returned)])
    assert registry['parent']['usable']
    assert registry['parent']['tissue_status_source'] == 'protocol-format-unknown'


@pytest.mark.parametrize('raw', ['', '<answer></answer>', '{"tissue_status":"present"}',
                                '<tissue_status>absent</tissue_status>'])
def test_unknown_status_fallback_does_not_invent_missing_observations(monkeypatch, raw):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    _mock_perceptor(backend, monkeypatch, [raw, raw])
    with pytest.raises(ModelProtocolError):
        backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert backend.tissue_status == {}


def test_unknown_status_fallback_does_not_accept_truncated_prose(monkeypatch):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    _mock_perceptor(backend, monkeypatch, ['Visible tissue.', 'Visible tissue.'])
    backend.generation['question_description_tokens'] = 1
    with pytest.raises(ModelProtocolError):
        backend.describe_image(Image.new('RGB', (8, 8)), 'Region=parent;', 'Which?')
    assert backend.tissue_status == {}


class WordTokenizer:
    def encode(self, text):
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(message['content'] for message in messages)


def test_summary_cannot_modify_authoritative_metadata():
    backend = StubBackend('R1_EVIDENCE', [])
    backend.config['algorithm']['summary_threshold'] = 0
    backend.tokenizer = WordTokenizer()
    backend._text = lambda *args, **kwargs: 'Region parent supposedly at 40x.'
    rows = [finding()]
    rows[0]['description'] = ' '.join(['long original description'] * 20)
    text = backend.evidence_text(rows, 'Which?')
    registry_json = text.split('\n', 1)[1].split('\n\nObserved descriptions:', 1)[0]
    metadata = json.loads(registry_json)
    assert metadata[0]['scale'] == 5
    assert metadata[0]['region_id'] == 'parent'
    assert 'supposedly at 40x' in text


def test_final_requires_valid_option_and_preserves_history_and_unknown():
    backend = StubBackend('R3_ROI_VALIDITY', [{'answer': 'Negative', 'explanation': 'invented'},
                                              {'answer': 'B', 'explanation': 'best estimate; insufficient tissue'}])
    trajectory = [{'iteration': 1, 'action': 'explore', 'decision': {'sufficient': 'No'}}]
    result = backend.conclude([finding(status='absent')], trajectory, SAMPLE)
    assert result['answer'] == 'B'
    assert 'unknown, not a negative label' in backend.records[-1]['messages'][0]['content']
    assert '"action": "explore"' in backend.records[-1]['messages'][1]['content']


def test_evidence_artifact_preserves_exact_pixels_and_deduplicates(tmp_path):
    from PIL import Image
    backend = StubBackend('R3_ROI_VALIDITY', [])
    backend.config['runtime'] = {'rules_v2_evidence_image_root': str(tmp_path)}
    image = Image.new('RGB', (19, 13), (71, 102, 193))
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    target = backend.save_evidence_image(image)
    assert target == str(tmp_path / (digest + '.png'))
    with Image.open(target) as restored:
        assert restored.mode == image.mode
        assert restored.size == image.size
        assert hashlib.sha256(restored.tobytes()).hexdigest() == digest
    assert backend.save_evidence_image(image) == target
    assert len(list(tmp_path.iterdir())) == 1
    assert image.size == (19, 13)
