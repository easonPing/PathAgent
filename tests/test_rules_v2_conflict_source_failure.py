"""Unknown observation sources fail R2 sufficiency without inventing a Region."""
import json
from copy import deepcopy

import pytest

from data_processing.regions import Region
from models.inference import ModelProtocolError
from models.rules_v2_inference import RulesV2ModelBackend
from models.rules_v2_prompts import flags_for_variant, templates_for_config


SAMPLE = {'question': 'Determine the molecular subtype from H&E.',
          'choices': {'A': 'Triple negative', 'B': 'Luminal A', 'C': 'HER2(+)', 'D': 'Luminal B'}}
KNOWN = '0_8192_m5/z40.0/4_7'
ACTUAL_OTHER_PARENT = '8192_0_m5/z40.0/6_2'
UNKNOWN = '0_8192_m5/z40.0/6_2'
PREDICTION = {
    'answer': 'A. Triple negative', 'thinking_steps': 'A morphological estimate with uncertainty.',
    'alternative_answer': 'D. Luminal B',
    'discriminating_feature': 'Nuclear pleomorphism and tubule formation',
    'conflicts': [
        {'region_id': KNOWN, 'feature': 'Nuclear pleomorphism',
         'claims': ['Marked nuclear pleomorphism', 'Moderate nuclear pleomorphism']},
        {'region_id': UNKNOWN, 'feature': 'Tubule formation',
         'claims': ['No definitive tubule formation', 'Focal loss of polarity']},
    ],
}
MISSING = {'missing_info': 'Tubule formation', 'zoom_recommendation': 'No',
           'recommended_zoom_level': None, 'zoom_reason': 'Explore representative morphology.'}


class SourceBackend(RulesV2ModelBackend):
    def __init__(self, responses, variant='R2_DIFFERENTIAL'):
        self.config = {'rules_v2': {'variant': variant}}
        self.rule_flags = flags_for_variant(variant)
        self.rule_templates = templates_for_config(self.config)
        self.tissue_status = {}
        self.generation = {'step_a_tokens': 1024, 'step_b_tokens': 512, 'step_c_tokens': 512,
                           'retries': 1}
        self.responses = deepcopy(responses)
        self.records = []

    def _evidence_for_v2(self, *args, **kwargs):
        return 'Trusted actual Region registry and source-separated observations.'

    def _text(self, messages, tokens, stage, greedy=False):
        raw = json.dumps(self.responses.pop(0))
        self.records.append({'stage': stage, 'messages': deepcopy(messages), 'raw_response': raw})
        return raw


def findings():
    return [{'region': Region(name, '/unused', x, y, 512, 512, 1024, 1024,
                              'physical', 40, .5, 'assumed', name.split('/')[0]).to_dict(),
             'description': 'Observed tissue.'}
            for name, x, y in ((KNOWN, 3584, 10240), (ACTUAL_OTHER_PARENT, 9216, 3072))]


def evaluate(backend):
    return backend.evaluate(findings(), SAMPLE, 'physical', [10, 20, 40])


def test_real_6654_parent_transposition_forces_no_without_retry_or_source_repair():
    backend = SourceBackend([PREDICTION, {'sufficient': 'Yes', 'unresolved_conflicts': []}, MISSING])
    decision = evaluate(backend)
    assert decision['answer'] == 'A. Triple negative'
    assert decision['alternative_answer'] == 'D. Luminal B'
    assert decision['conflicts'] == PREDICTION['conflicts']
    assert decision['reflection_raw'] == {'sufficient': 'Yes', 'unresolved_conflicts': []}
    assert decision['sufficient'] == decision['effective_sufficient'] == 'No'
    assert decision['evidence_failures'] == [
        {'reason': 'invalid_conflict_source', 'region_id': UNKNOWN, 'feature': 'Tubule formation',
         'claims': ['No definitive tubule formation', 'Focal loss of polarity']}]
    assert [r['stage'] for r in backend.records] == ['predict', 'reflect', 'missing_info']
    assert json.loads(backend.records[0]['raw_response']) == PREDICTION
    missing_input = backend.records[-1]['messages'][1]['content']
    assert 'invalid_conflict_source' in missing_input and UNKNOWN in missing_input
    assert decision['zoom_recommendation'] == 'No' and decision['zoom_level'] is None


@pytest.mark.parametrize('bad_region', [None, '', ' ', 4, [], {}])
def test_conflict_region_remains_strict_nonempty_string_structure(bad_region):
    prediction = deepcopy(PREDICTION)
    prediction['conflicts'][1]['region_id'] = bad_region
    backend = SourceBackend([prediction, prediction])
    with pytest.raises(ModelProtocolError, match='Invalid predict response after 2 attempts'):
        evaluate(backend)
    assert len(backend.records) == 2


def test_unknown_source_and_unlisted_primary_both_remain_evidence_failures():
    invalid = {**PREDICTION, 'answer': 'Cannot determine'}
    backend = SourceBackend([invalid, {'sufficient': 'Yes', 'unresolved_conflicts': []}, MISSING])
    result = evaluate(backend)
    assert result['answer'] == 'Cannot determine' and result['sufficient'] == 'No'
    assert {failure['reason'] for failure in result['evidence_failures']} >= {
        'invalid_primary_answer', 'invalid_conflict_source'}
    assert [record['stage'] for record in backend.records] == ['predict', 'reflect', 'missing_info']


def test_known_opposing_claims_keep_existing_reflection_decision():
    prediction = deepcopy(PREDICTION)
    prediction['conflicts'][1]['region_id'] = ACTUAL_OTHER_PARENT
    backend = SourceBackend([prediction, {'sufficient': 'Yes', 'unresolved_conflicts': []}])
    decision = evaluate(backend)
    assert decision['sufficient'] == 'Yes'
    assert decision['evidence_failures'] == []
    assert len(backend.records) == 2


@pytest.mark.parametrize('variant', ['R1_EVIDENCE', 'R3_ROI_VALIDITY',
                                   'R4_VISUAL_MISSING', 'R5_TARGET_REGION'])
def test_unknown_conflict_sufficiency_gate_is_only_r2(variant):
    backend = SourceBackend([], variant)
    prediction = deepcopy(PREDICTION)
    reflection = {'sufficient': 'Yes'}
    if variant == 'R1_EVIDENCE':
        prediction['required_features'] = [{'feature': 'nuclear detail', 'evidence_type': 'cellular'}]
        reflection['checks'] = [{'feature': 'nuclear detail', 'evidence_type': 'cellular',
                                 'region_ids': [KNOWN], 'assessable': True,
                                 'coverage_sufficient': True}]
    if variant == 'R4_VISUAL_MISSING':
        reflection['target_observed'] = 'Unknown'
    sufficient, failures, _, _ = backend._effective_reflection(
        prediction, reflection, backend._registry(findings()), SAMPLE['choices'])
    assert sufficient == 'Yes' and failures == []
