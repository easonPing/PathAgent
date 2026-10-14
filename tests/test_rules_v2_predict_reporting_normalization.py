"""Retry-only normalization of an omitted reporting field; clinical fields stay strict."""
from copy import deepcopy
import json

import pytest

from models.inference import ModelProtocolError
from models.rules_v2_inference import RulesV2ModelBackend
from models.rules_v2_prompts import flags_for_variant, predict_keys, templates_for_config


CHOICES = {'A': '0', 'B': '2+', 'C': '3+', 'D': '1+'}
REGIONS = ('20480_0_m5', '28672_4096_m5', '20480_0_m5/z40.0/0_2',
           '20480_0_m5/z40.0/0_6', '20480_0_m5/z40.0/0_3', '20480_0_m5/z40.0/0_5')
# Exact field values from the second Predict response of real R2 bcnb:4245.
REAL_RETRY = {
    'answer': 'B', 'alternative_answer': 'D',
    'conflicts': [
        {'claims': claims, 'feature': 'HER2 expression level', 'region_id': region}
        for region, claims in zip(REGIONS,
                                 (['0', '2+'], ['0', '3+'], ['1+', '2+'],
                                  ['1+', '3+'], ['1+', '2+'], ['1+', '3+']))
    ],
    'discriminating_feature': 'HER2 expression level cannot be determined from H&E images; '
                             'it requires immunohistochemical (IHC) analysis to evaluate '
                             'membranous staining intensity and completeness.',
}


class ReportingBackend(RulesV2ModelBackend):
    def __init__(self, responses):
        self.config = {'rules_v2': {'variant': 'R2_DIFFERENTIAL'}}
        self.rule_flags = flags_for_variant('R2_DIFFERENTIAL')
        self.rule_templates = templates_for_config(self.config)
        self.generation = {'retries': 1}
        self.responses = deepcopy(responses)
        self.records = []

    def _text(self, messages, tokens, stage, greedy=False):
        raw = json.dumps(self.responses.pop(0))
        self.records.append({'stage': stage, 'messages': deepcopy(messages), 'raw_response': raw})
        return raw


def parse(backend):
    return backend._json('system', 'prompt', 1024, 'predict', predict_keys(backend.rule_flags),
                         validator=lambda value: backend._valid_predict(value, dict.fromkeys(REGIONS), CHOICES),
                         retry_context={'choices': CHOICES})


def test_real_4245_retry_reporting_omission_preserves_all_clinical_fields_and_raw():
    backend = ReportingBackend([{'answer': 'No'}, REAL_RETRY])
    result = parse(backend)
    assert result == {**REAL_RETRY, 'thinking_steps': ''}
    assert len(backend.records) == 2
    record = backend.records[-1]
    assert json.loads(record['raw_response']) == record['parsed_raw_response'] == REAL_RETRY
    assert 'thinking_steps' not in record['parsed_raw_response']
    assert record['value_normalization'] == {
        'thinking_steps': {'from': '<omitted>', 'to': '', 'reason': 'reporting_field_omitted_on_retry'}}
    assert 'thinking_steps' in record['messages'][-1]['content']


@pytest.mark.parametrize('change', [
    {'alternative_answer': None}, {'discriminating_feature': ''},
    {'conflicts': [{'region_id': REGIONS[0], 'feature': 'HER2', 'claims': ['one claim']}]},
    {'thinking_steps': None},
])
def test_retry_reporting_normalization_does_not_relax_answer_or_rule_fields(change):
    invalid = {**REAL_RETRY, **change}
    backend = ReportingBackend([{'answer': 'No'}, invalid])
    with pytest.raises(ModelProtocolError, match='Invalid predict response after 2 attempts'):
        parse(backend)
    assert len(backend.records) == 2
    assert backend.records[-1]['raw_response'] == json.dumps(invalid)
    if 'thinking_steps' in change:
        assert 'value_normalization' not in backend.records[-1]


def test_first_attempt_missing_reporting_field_still_retries_and_existing_text_is_retained():
    valid = {**REAL_RETRY, 'thinking_steps': 'actual model-provided explanation'}
    backend = ReportingBackend([REAL_RETRY, valid])
    result = parse(backend)
    assert len(backend.records) == 2
    assert result == valid
    assert all('value_normalization' not in record for record in backend.records)


def test_missing_clinical_rule_field_is_not_added_by_reporting_normalization():
    invalid = deepcopy(REAL_RETRY)
    invalid.pop('conflicts')
    backend = ReportingBackend([{'answer': 'No'}, invalid])
    with pytest.raises(ModelProtocolError):
        parse(backend)
    assert 'conflicts' not in backend.records[-1]['parsed_raw_response']
