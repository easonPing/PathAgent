"""An unlisted primary estimate is insufficient; only the model may pick Final."""
from copy import deepcopy

import pytest

from models.inference import ModelProtocolError
from test_rules_v2_conflict_source_failure import SourceBackend, MISSING, KNOWN, findings


SAMPLE = {'question': 'Estimate HER2 status from H&E.', 'choices': {'A': 'Negative', 'B': 'Positive'}}
REFUSAL = 'The HER2 status cannot be determined from the provided H&E image.'


def prediction(answer=REFUSAL, alternative=REFUSAL):
    return {'answer': answer, 'alternative_answer': alternative,
            'thinking_steps': 'IHC staining is unavailable.',
            'discriminating_feature': 'nuclear morphology', 'conflicts': []}


def run(backend):
    return backend.evaluate(findings(), SAMPLE, 'physical', [10, 20, 40])


def test_actual_3818_primary_refusal_cannot_pass_model_yes_and_enters_missing():
    original = prediction()
    backend = SourceBackend([original, {'sufficient': 'Yes', 'unresolved_conflicts': []}, MISSING,
                             {'answer': 'B. Positive', 'explanation': 'A limited indirect estimate.'}])
    decision = run(backend)
    assert decision['answer'] == decision['alternative_answer'] == REFUSAL
    assert decision['thinking_steps'] == original['thinking_steps']
    assert decision['reflection_raw']['sufficient'] == 'Yes'
    assert decision['sufficient'] == 'No'
    assert decision['evidence_failures'] == [
        {'reason': 'invalid_primary_answer', 'primary_answer': REFUSAL},
        {'reason': 'invalid_differential', 'primary_answer': REFUSAL, 'alternative_answer': REFUSAL}]
    assert [r['stage'] for r in backend.records] == ['predict', 'reflect', 'missing_info']
    assert REFUSAL in backend.records[-1]['messages'][1]['content']
    assert 'invalid_primary_answer' in backend.records[-1]['messages'][1]['content']
    backend.generation['final_tokens'] = 2048
    final = backend.conclude(findings(), [{'iteration': 1, 'action': 'explore', 'decision': decision}], SAMPLE)
    assert final['answer'] == 'B. Positive'  # The fixture model selected it; no default Negative.
    assert decision['answer'] == REFUSAL


def test_listed_alternative_does_not_replace_unlisted_primary():
    backend = SourceBackend([prediction('Uncertain', 'B. Positive'),
                             {'sufficient': 'Yes', 'unresolved_conflicts': []}, MISSING])
    decision = run(backend)
    assert decision['answer'] == 'Uncertain' and decision['alternative_answer'] == 'B. Positive'
    assert decision['evidence_failures'] == [{'reason': 'invalid_primary_answer', 'primary_answer': 'Uncertain'}]
    assert decision['sufficient'] == 'No'


def test_all_2608_uncertain_primary_keeps_required_features_and_source_contract():
    original = prediction('Uncertain', 'Uncertain')
    original['required_features'] = [{'feature': 'PR_staining_intensity', 'evidence_type': 'fine_detail'}]
    reflected = {'sufficient': 'Yes', 'unresolved_conflicts': [], 'target_observed': 'Unknown',
                 'checks': [{'feature': 'PR_staining_intensity', 'evidence_type': 'fine_detail',
                             'region_id': KNOWN, 'assessable': True, 'coverage_sufficient': True}]}
    missing = {**MISSING, 'target_region': None, 'evidence_type': 'fine_detail'}
    backend = SourceBackend([original, reflected, missing], 'ALL5')
    decision = run(backend)
    assert decision['answer'] == 'Uncertain' and decision['prediction_raw'] == original
    assert decision['required_features'] == original['required_features']
    assert not decision['scale_failures'] and decision['sufficient'] == 'No'
    assert {f['reason'] for f in decision['evidence_failures']} == {
        'invalid_primary_answer', 'invalid_differential'}


@pytest.mark.parametrize('answer', [None, '', ' ', 5, [], {}])
def test_primary_structure_still_requires_nonempty_string_and_retries(answer):
    invalid = prediction(answer)
    backend = SourceBackend([invalid, invalid])
    with pytest.raises(ModelProtocolError, match='Invalid predict response after 2 attempts'):
        run(backend)
    assert len(backend.records) == 2


def test_required_clinical_rule_field_omission_remains_format_failure():
    invalid = prediction()
    invalid.pop('conflicts')
    backend = SourceBackend([invalid, invalid])
    with pytest.raises(ModelProtocolError):
        run(backend)
    assert all('value_normalization' not in r for r in backend.records)


def test_final_refusal_is_still_invalid_after_retry_and_never_gets_default_label():
    refused = {'answer': REFUSAL, 'explanation': 'IHC is unavailable.'}
    backend = SourceBackend([refused, deepcopy(refused)])
    backend.generation['final_tokens'] = 2048
    with pytest.raises(ModelProtocolError, match='Invalid final response after 2 attempts'):
        backend.conclude(findings(), [], SAMPLE)
    assert [r['stage'] for r in backend.records] == ['final', 'final']
