"""A failed R2 Reflect format cannot support Yes or masquerade as model No."""
from copy import deepcopy
import json

import pytest

from models.inference import ModelProtocolError
from test_rules_v2_conflict_source_failure import SourceBackend, SAMPLE, PREDICTION, MISSING, UNKNOWN, findings


TRUNCATED_3702 = '{"sufficient": "No", "unresolved_conflicts": [{"region_id": "24576_24576_m5/z40.0/0_3", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/0_4", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/0_5", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/0_6", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/1_3", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/1_4", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/1_5", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/1_6", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/1_7", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/2_3", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/2_4", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/5_5", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/5_6", "feature": "HER2 status"}, {"region_id": "24576_24576_m5/z40.0/5_7", "feature":'


class WireBackend(SourceBackend):
    def _text(self, messages, tokens, stage, greedy=False):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        raw = response if isinstance(response, str) else json.dumps(response)
        self.records.append({'stage': stage, 'messages': deepcopy(messages), 'raw_response': raw,
                             'at_token_cap': raw in (TRUNCATED_3702, TRUNCATED_3702.replace('"No"', '"Yes"', 1)) or raw in TRUNCATED_R1_6654})
        return raw


def run(backend):
    return backend.evaluate(findings(), SAMPLE, 'physical', [10, 20, 40])


def test_actual_3702_two_truncations_are_unknown_and_keep_raw_conflicts_until_final():
    backend = WireBackend([PREDICTION, TRUNCATED_3702, TRUNCATED_3702, MISSING,
                           {'answer': 'A', 'explanation': 'Indirect estimate; evidence remains limited.'}])
    decision = run(backend)
    assert decision['reflection_raw'] is None
    assert decision['reflection_source'] == 'protocol-format-unknown'
    assert decision['sufficient'] == decision['effective_sufficient'] == 'No'
    assert decision['conflicts'] == PREDICTION['conflicts']
    assert decision['unresolved_conflicts'] == [
        {'region_id': c['region_id'], 'feature': c['feature']} for c in PREDICTION['conflicts']]
    assert any(f['reason'] == 'invalid_conflict_source' and f['region_id'] == UNKNOWN
               for f in decision['evidence_failures'])
    assert decision['reflection_protocol_failure'] == {
        'reason': 'Invalid reflect response after 2 attempts',
        'attempts': [{'record_index': i, 'raw_response': TRUNCATED_3702, 'at_token_cap': True}
                     for i in (1, 2)]}
    assert [r['stage'] for r in backend.records] == ['predict', 'reflect', 'reflect', 'missing_info']
    assert backend.records[1]['raw_response'] == backend.records[2]['raw_response'] == TRUNCATED_3702
    assert all('parsed_raw_response' not in r for r in backend.records[1:3])
    assert 'protocol-format-unknown' in backend.records[-1]['messages'][1]['content']
    backend.generation['final_tokens'] = 2048
    final = backend.conclude(findings(), [{'iteration': 1, 'action': 'explore', 'decision': decision}], SAMPLE)
    assert final['answer'] == 'A' and 'limited' in final['explanation']


def test_truncated_yes_prefix_never_becomes_model_yes_or_model_no():
    raw = TRUNCATED_3702.replace('"No"', '"Yes"', 1)
    backend = WireBackend([PREDICTION, raw, raw, MISSING])
    decision = run(backend)
    assert decision['reflection_raw'] is None and decision['sufficient'] == 'No'
    assert all('"sufficient": "Yes"' in attempt['raw_response']
               for attempt in decision['reflection_protocol_failure']['attempts'])


def test_all_failure_supplies_no_assessability_and_unknown_progress_only():
    prediction = {**PREDICTION, 'required_features': [{'feature': 'nuclear detail', 'evidence_type': 'cellular'}]}
    missing = {**MISSING, 'target_region': None, 'evidence_type': 'cellular'}
    backend = WireBackend([prediction, TRUNCATED_3702, TRUNCATED_3702, missing], 'ALL5')
    decision = run(backend)
    assert decision['checks'] == []
    assert decision['target_observed'] == 'Unknown'
    assert decision['reflection_raw'] is None and decision['sufficient'] == 'No'
    assert any(f['reason'] == 'missing_or_duplicate_check' for f in decision['evidence_failures'])
    assert not decision['scale_failures']  # No fabricated citation/ROI.


@pytest.mark.parametrize('variant', ['R3_ROI_VALIDITY', 'R4_VISUAL_MISSING', 'R5_TARGET_REGION'])
def test_conservative_format_fallback_is_isolated_to_r1_r2_and_all(variant):
    prediction = deepcopy(PREDICTION)
    if variant == 'R1_EVIDENCE':
        prediction['required_features'] = [{'feature': 'nuclear detail', 'evidence_type': 'cellular'}]
    backend = WireBackend([prediction, TRUNCATED_3702, TRUNCATED_3702], variant)
    with pytest.raises(ModelProtocolError, match='Invalid reflect response after 2 attempts'):
        run(backend)
    assert len(backend.records) == 3


@pytest.mark.parametrize('error', [ModelProtocolError('Executor prompt exceeds context limit'),
                                 ModelProtocolError('Summary did not reduce overlong evidence'),
                                 OSError('I/O failure'), RuntimeError('CUDA out of memory')])
def test_context_io_and_cuda_errors_are_not_swallowed(error):
    backend = WireBackend([PREDICTION, error])
    with pytest.raises(type(error), match=str(error)):
        run(backend)
    assert len(backend.records) == 1


def test_error_text_without_two_actual_reflect_records_is_not_format_fallback():
    backend = WireBackend([PREDICTION, ModelProtocolError('Invalid reflect response after 2 attempts')])
    with pytest.raises(ModelProtocolError):
        run(backend)
    assert len(backend.records) == 1


def test_valid_model_reflection_keeps_original_contract_without_new_source_fields():
    prediction = {**PREDICTION, 'conflicts': []}
    reflected = {'sufficient': 'Yes', 'unresolved_conflicts': []}
    backend = WireBackend([prediction, reflected])
    decision = run(backend)
    assert decision['reflection_raw'] == reflected
    assert decision['sufficient'] == 'Yes'
    assert 'reflection_source' not in decision and 'reflection_protocol_failure' not in decision
    assert [r['stage'] for r in backend.records] == ['predict', 'reflect']


# Exact raw outputs of actual R1 bcnb:6654 calls 10 and 11.
TRUNCATED_R1_6654 = ('{\n  "sufficient": "Yes",\n  "checks": [\n    {\n      "feature": "HER2 staining intensity",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/0_1",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining proportion",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/0_1",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining localization",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/0_1",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "Normal tissue HER2 status",\n      "evidence_type": "overview",\n      "region_id": "0_8192_m5",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining intensity",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/1_2",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining proportion",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/1_2",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining localization",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/1_2",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "Normal tissue HER2 status",\n      "evidence_type": "overview",\n      "region_id": "0_8192_m5",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "', '{\n  "sufficient": "Yes",\n  "checks": [\n    {\n      "feature": "HER2 staining intensity",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/0_1",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining proportion",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/0_1",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining localization",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/0_1",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "Normal tissue HER2 status",\n      "evidence_type": "overview",\n      "region_id": "0_8192_m5",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining intensity",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/1_2",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining proportion",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/1_2",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "HER2 staining localization",\n      "evidence_type": "fine_detail",\n      "region_id": "0_8192_m5/z40.0/1_2",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "feature": "Normal tissue HER2 status",\n      "evidence_type": "overview",\n      "region_id": "0_8192_m5",\n      "assessable": true,\n      "coverage_sufficient": true\n    },\n    {\n      "')
R1_6654_PREDICTION = {'answer': 'C. HER2(+)', 'thinking_steps': 'The image shows HER2 immunohistochemistry (IHC) staining with strong, complete membranous staining in >10% of tumor cells, meeting criteria for HER2 positivity. This confirms a HER2-positive molecular subtype, which is characterized by HER2 overexpression. The other options (triple negative, luminal A, luminal B) are not supported by the evidence in the image, which specifically demonstrates HER2 overexpression.', 'required_features': [{'feature': 'HER2 staining intensity', 'evidence_type': 'fine_detail'}, {'feature': 'HER2 staining proportion', 'evidence_type': 'fine_detail'}, {'feature': 'HER2 staining localization', 'evidence_type': 'fine_detail'}, {'feature': 'Normal tissue HER2 status', 'evidence_type': 'overview'}]}


def test_actual_r1_6654_duplicate_check_truncation_produces_unknown_no_without_conflicts():
    backend = WireBackend([R1_6654_PREDICTION, *TRUNCATED_R1_6654, MISSING], 'R1_EVIDENCE')
    decision = run(backend)
    assert decision['reflection_raw'] is None and decision['sufficient'] == 'No'
    assert decision['reflection_source'] == 'protocol-format-unknown'
    assert decision['checks'] == []
    assert 'unresolved_conflicts' not in decision
    assert 'conflicts' not in decision
    assert [attempt['raw_response'] for attempt in decision['reflection_protocol_failure']['attempts']] == list(TRUNCATED_R1_6654)
    assert all(attempt['at_token_cap'] for attempt in decision['reflection_protocol_failure']['attempts'])
    assert not decision['scale_failures']
    assert [record['stage'] for record in backend.records] == ['predict', 'reflect', 'reflect', 'missing_info']
