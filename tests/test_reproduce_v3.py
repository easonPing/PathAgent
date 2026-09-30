"""CPU-only regressions; these do not run validation/benchmark inference."""
import copy
import csv
import json

import pytest
from PIL import Image

from data_processing.common import load_config, run_identity
from data_processing.datasets import load_cptac, public_input
from data_processing.regions import Region
from models.context_inference import ContextModelBackend
from models.paper_prompts import RETRY_USER
from models.protocols import CONTEXT_PROTOCOL, backend_class, protocol_manifest, terminal_status
from scripts.prepare_bcnb_sample import stratified_sample
from test_protocols import scripted_backend, generative_backend

V3 = 'configs/reproduce_v3.yaml'


def test_v3_changes_only_protocol_and_cache():
    v2, v3 = map(load_config, ['configs/reproduce_v2.yaml', V3])
    assert backend_class(v3) is ContextModelBackend
    assert v3['protocol'] == CONTEXT_PROTOCOL
    assert v3['runtime']['processed_root'] == 'data/processed/reproduce-v3/bf16'
    assert terminal_status(v3, 'invalid')
    assert not terminal_status(v3, 'error')
    assert protocol_manifest(v3)['protocol'] == CONTEXT_PROTOCOL
    assert run_identity(v2, 'source', 'samples') != run_identity(v3, 'source', 'samples')
    v3['protocol'] = v2['protocol']
    v3['runtime']['processed_root'] = v2['runtime']['processed_root']
    assert v3 == v2


@pytest.mark.parametrize('question,missing', [(None, None), ('Which?', None), ('Which?', ''), ('Which?', '核分裂 "detail"\nmore')])
def test_actual_perceptor_messages_and_logs(question, missing, monkeypatch):
    backend, calls, messages, _ = generative_backend(V3, monkeypatch, '<think>raw</think>description')
    region = Region('one', '/unused', 10, 20, 4096, 4096, 512, 512, 'physical', 5, .5)
    with Image.new('RGB', (16, 16)) as image:
        assert backend.describe_image(image, region.metadata(), question, missing) == '<think>raw</think>description'
    actual = messages[0][0][1]['content'][1]['text']
    assert actual.startswith(region.metadata() + '\n')
    assert backend.records[-1]['messages'][1]['content'][1]['text'] == actual
    assert backend.records[-1]['prompt'] == actual
    assert ('Missing visual information to inspect:' in actual) == bool(missing)
    if missing:
        assert actual.endswith(missing)
    if question:
        assert question in actual
    assert 'do_sample' not in calls[0]


def test_missing_receives_current_prediction_and_actual_reflection():
    a = {'answer': 'A', 'thinking_steps': '中文 "quoted"\nnext line'}
    b = {'sufficient': 'No'}
    c = {'missing_info': 'cells', 'zoom_recommendation': 'Yes',
         'recommended_zoom_level': '40', 'zoom_reason': 'detail'}
    backend, calls = scripted_backend(V3, [json.dumps(x) for x in (a, b, c, {**a, 'answer': 'B'}, b, c)])
    row = {'question': 'Which?', 'choices': {'A': 'one', 'B': 'two'}, 'ground_truth': 'SECRET'}
    for answer in ('A', 'B'):
        decision = backend.evaluate([], row, 'physical', [10, 20, 40])
        assert decision['zoom_level'] == 40
        prompt = calls[-1]['messages'][1]['content']
        prediction = prompt.split('Previous answer and reasoning:\n')[1].split('\n\nPrevious reflection:')[0]
        reflection = prompt.split('Previous reflection:\n')[1].split('\n\nNow provide')[0]
        assert json.loads(prediction) == {**a, 'answer': answer}
        assert json.loads(reflection) == b
        assert 'SECRET' not in prompt


def test_sufficient_skips_missing():
    backend, calls = scripted_backend(V3, ['{"answer":"A","thinking_steps":"why"}', '{"sufficient":"Yes"}'])
    backend.evaluate([], {'question': 'Which?', 'choices': {}}, 'physical', [10, 20, 40])
    assert [c['stage'] for c in calls] == ['predict', 'reflect']


def test_v3_budget_preserves_prior_outputs_and_retry_while_compressing_evidence():
    backend = object.__new__(ContextModelBackend)
    backend.config = load_config(V3)
    backend.generation = backend.config['generation']
    backend.context_limit = 6000
    envelopes, summaries = [], []
    class Tokenizer:
        def encode(self, text): return list(text)
        def apply_chat_template(self, messages, **kwargs):
            envelopes.append(copy.deepcopy(messages))
            return ''.join(m['content'] for m in messages)
    backend.tokenizer = Tokenizer()
    def summarize(messages, tokens, stage, greedy=False):
        assert stage == 'summary'
        summaries.append(messages)
        return 'compressed evidence'
    backend._text = summarize
    prediction = {'answer': 'A', 'thinking_steps': '理由"\n' * 200}
    reflection = {'sufficient': 'No'}
    render = lambda d: backend.missing_user(d, {'question': 'Which?', 'choices': ''}, 'absolute magnification', [10, 20, 40], prediction, reflection)
    region = Region('one', '/unused', 0, 0, 40, 40, 10, 10, 'physical', 5, .5)
    result = backend._evidence_for([{'region': region.to_dict(), 'description': 'x' * 5000}], 'system', render, 512, 'Which?')
    assert result == 'compressed evidence' and len(summaries) == 1
    for messages in envelopes:
        assert json.dumps(prediction, ensure_ascii=False) in messages[1]['content']
        assert json.dumps(reflection, ensure_ascii=False) in messages[1]['content']
        assert messages[-1]['content'] == RETRY_USER


def test_sampling_covers_rare_semantic_classes_and_is_order_independent():
    rows = [{'ID': str(i), 'Task': 'Tumor', 'Answer': 'A' if i % 2 else 'B',
             'A': 'common' if i % 2 else 'rare' if i == 0 else 'unused',
             'B': 'rare' if i == 0 else 'common'} for i in range(100)]
    selected, strata = stratified_sample(rows, 10, 128)
    assert len(selected) == len({r['ID'] for r in selected}) == 10
    assert all(s['sample_count'] >= 1 for s in strata)
    assert len(strata) == 2
    assert stratified_sample(list(reversed(rows)), 10, 128) == (selected, strata)


def test_cptac_never_imports_released_predictions(tmp_path):
    path = tmp_path / 'cptac.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Slide', 'Tumor', 'Task', 'Question', 'A', 'B', 'Answer', 'Output'])
        writer.writeheader()
        writer.writerow(dict(ID='1', Slide='slide', Tumor='LUAD', Task='', Question='Which?', A='one', B='two', Answer='A', Output='RELEASED_PREDICTION'))
    rows = load_cptac(path, tmp_path, require_images=False)
    assert rows[0]['ground_truth_label'] == 'A'
    assert rows[0]['task'] == 'Tumor subtype'
    assert 'RELEASED_PREDICTION' not in json.dumps(rows)
    assert 'ground_truth' not in public_input(rows[0])
    with pytest.raises(ValueError, match='expected one original'):
        load_cptac(path, tmp_path, require_images=True)
