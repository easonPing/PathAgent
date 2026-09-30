"""CPU regression against fixed legacy/upstream evidence, plus runner integration."""
import copy
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from data_processing.common import (atomic_json, digest, load_config, read_json,
                                    run_identity, write_manifest)
from data_processing.regions import Region
from models.inference import ModelBackend, ModelProtocolError, parse_json_object as legacy_json
from models.paper_inference import PaperModelBackend, parse_json_object as paper_json
from models.protocols import (LEGACY_PROTOCOL, PAPER_PROTOCOL, backend_class,
                              protocol_manifest, protocol_name)

FIXTURES = Path(__file__).parent / 'fixtures'
CONFIGS = ['configs/reproduce_v1.yaml', 'configs/reproduce_v2.yaml']


def scripted_backend(config_path, outputs):
    cfg = load_config(config_path)
    backend = object.__new__(backend_class(cfg))
    backend.config, backend.generation = cfg, cfg['generation']
    backend.evidence_text = lambda *args, **kwargs: 'visible evidence'
    backend._evidence_for = lambda *args, **kwargs: 'visible evidence'
    answers, calls = iter(outputs), []
    def text(messages, tokens, stage, greedy=False):
        calls.append({'messages': copy.deepcopy(messages), 'tokens': tokens,
                      'stage': stage, 'greedy': greedy})
        return next(answers)
    backend._text = text
    return backend, calls


def test_local_configs_share_environment_sampling_and_algorithm():
    old, new = map(load_config, CONFIGS)
    assert backend_class(old) is ModelBackend
    assert backend_class(new) is PaperModelBackend
    assert protocol_name(old) == LEGACY_PROTOCOL
    for key in ['models', 'algorithm', 'preprocessing', 'slurm', 'seed']:
        assert old[key] == new[key]
    assert old['runtime']['workers'] == new['runtime']['workers'] == 1
    assert old['runtime']['processed_root'] != new['runtime']['processed_root']
    for key, value in old['generation'].items():
        assert new['generation'][key] == (value * 2 if key.endswith('_tokens') else value)
    assert new['generation']['perceptor_sampling'] == 'checkpoint'


@pytest.mark.parametrize('change', ['unknown', 'budget', 'sampling', 'quantization', 'devices'])
def test_config_rejects_unsupported_protocol_changes(tmp_path, change):
    cfg = load_config(CONFIGS[1])
    if change == 'unknown': cfg['protocol'] = 'typo'
    if change == 'budget': cfg['generation']['final_tokens'] = 4096
    if change == 'sampling': cfg['generation']['perceptor_sampling'] = 'greedy'
    if change == 'quantization': cfg['models']['executor']['quantization'] = '4bit'
    if change == 'devices': cfg['runtime']['devices'] = {'executor': 'cuda:1'}
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError): load_config(path)


def test_paper_prompt_texts_match_pinned_github_commit():
    from models.paper_prompts import prompt_manifest
    reference = read_json(FIXTURES / 'paper_protocol_1e4d037.json')
    assert prompt_manifest() == reference['prompts']
    actual = protocol_manifest(load_config(CONFIGS[1]))
    assert actual['upstream_commit'] == reference['upstream_commit']
    assert actual['perceptor_sampling'] == 'checkpoint'


@pytest.mark.parametrize('config,fixture', zip(CONFIGS, ['legacy_calls_fe18766.json', 'paper_calls_1e4d037.json']))
def test_executor_messages_and_budgets_match_reference(config, fixture):
    backend, calls = scripted_backend(config, [
        '{"answer":"A","thinking_steps":"structured reasoning"}', '{"sufficient":"No"}',
        json.dumps({'missing_info':'mitoses', 'zoom_recommendation':'Yes', 'zoom_level':40,
                    'recommended_zoom_level':40, 'zoom_reason':'detail'}),
        '{"answer":"A","explanation":"all evidence"}',
    ])
    row = {'question':'Which?', 'choices':{'A':'yes', 'B':'no'}}
    decision = backend.evaluate([], row, 'physical', [10,20,40])
    backend.conclude([], [{'iteration':1, 'action':'zoom', 'decision':decision}], row)
    assert calls == read_json(FIXTURES / fixture)['calls']


@pytest.mark.parametrize('config', CONFIGS)
def test_sufficient_skips_missing_and_retry_is_bounded(config):
    backend, calls = scripted_backend(config, [
        '{"answer":"A","thinking_steps":"evidence"}', '{"sufficient":"Yes"}',
        'bad', 'bad'])
    row = {'question':'Which?', 'choices':{'A':'yes','B':'no'}}
    backend.evaluate([], row, 'physical', [10,20,40])
    assert [c['stage'] for c in calls] == ['predict','reflect']
    with pytest.raises(ModelProtocolError): backend.conclude([], [], row)
    assert len(calls) == 4
    assert calls[-1]['tokens'] == calls[-2]['tokens'] == backend.generation['final_tokens']
    assert calls[-1]['greedy'] is calls[-2]['greedy'] is True


@pytest.mark.parametrize('level,kind,allowed,valid', [
    (40, 'physical', [10,20,40], True), ('40', 'physical', [10,20,40], True),
    ('10', 'physical', [10,20,40], True), ('20', 'physical', [10,20,40], True),
    ('40x', 'physical', [10,20,40], False), ('40.0', 'physical', [10,20,40], False),
    (' 40', 'physical', [10,20,40], False), ('100', 'physical', [10,20,40], False),
    ('40', 'relative', [2,4,8], False), ('2', 'relative', [2,4,8], False),
    (100, 'physical', [10,20,40], False), (20, 'relative', [2,4,8], False),
    (2, 'relative', [2,4,8], True),
])
def test_paper_zoom_schema_normalizes_only_three_allowed_strings(level, kind, allowed, valid):
    answer = json.dumps({'missing_info':'cells', 'zoom_recommendation':'Yes',
                         'recommended_zoom_level':level, 'zoom_reason':'detail'})
    backend, _ = scripted_backend(CONFIGS[1], [
        '{"answer":"A","thinking_steps":"evidence"}', '{"sufficient":"No"}', answer, answer])
    row = {'question':'Which?', 'choices':{}}
    if valid:
        decision = backend.evaluate([], row, kind, allowed)
        assert decision['zoom_level'] == int(level)
        assert type(decision['recommended_zoom_level']) is int
    else:
        with pytest.raises(ModelProtocolError): backend.evaluate([], row, kind, allowed)


def test_think_parser_does_not_change_legacy():
    text = '<think>{"answer":"thought"}</think>{"answer":"final"}'
    assert legacy_json(text) == {'answer':'thought'}
    assert paper_json(text) == {'answer':'final'}
    assert paper_json('<think>{"answer":"unfinished"}') is None


class FakeInputs(dict):
    def __getattr__(self, key): return self[key]
    def to(self, device):
        assert device == 'cuda:0'
        return self


def generative_backend(config, monkeypatch, output='description', sample=True):
    """Exercise actual generate() call sites with CPU tensors and a fake checkpoint."""
    import torch
    import qwen_vl_utils
    cfg = load_config(config)
    backend = object.__new__(backend_class(cfg))
    backend.config, backend.generation = cfg, cfg['generation']
    backend.records, backend.context_limit = [], 32768
    backend._seed = lambda stage: 128
    calls, messages = [], []
    generation = {'do_sample':sample, 'temperature':.7, 'top_p':.8, 'top_k':19, 'eos_token_id':3}
    class Processor:
        image_processor = SimpleNamespace(merge_size=2)
        def apply_chat_template(self, value, **kwargs):
            messages.append((value,kwargs))
            return 'input'
        def __call__(self, *args, **kwargs):
            return FakeInputs(input_ids=torch.tensor([[1,2]]), image_grid_thw=torch.tensor([[1,4,4]]))
        def decode(self, *args, **kwargs): return output
    class Model:
        generation_config = SimpleNamespace(eos_token_id=3, to_dict=lambda: dict(generation))
        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.tensor([[1,2,7,3]])
    backend.processor = backend.tokenizer = Processor()
    backend.perceptor = backend.executor = Model()
    monkeypatch.setattr(qwen_vl_utils, 'process_vision_info', lambda msgs: ([msgs[1]['content'][0]['image']], None))
    return backend, calls, messages, generation


@pytest.mark.parametrize('config', CONFIGS)
@pytest.mark.parametrize('question,missing', [(None,None), ('Which?',None), ('Which?','mitoses')])
@pytest.mark.parametrize('sample', [True,False])
def test_perceptor_inherits_identical_checkpoint_sampling(config, question, missing, sample, monkeypatch):
    raw = '<think>observations</think>description'
    backend, calls, messages, generation = generative_backend(config, monkeypatch, raw, sample)
    image = Image.new('RGB', (16,16))
    assert backend.describe_image(image, 'region coordinates', question, missing) == raw
    passed = {k:v for k,v in calls[0].items() if k not in {'input_ids','image_grid_thw'}}
    expected_tokens = backend.generation['generic_description_tokens' if question is None else 'question_description_tokens']
    # No sampling override, even when the checkpoint requests sampling.
    assert passed == {'max_new_tokens': expected_tokens}
    content = messages[0][0][1]['content'][1]['text']
    if isinstance(backend, PaperModelBackend):
        assert 'region coordinates' not in content and 'mitoses' not in content
        assert backend.records[-1]['generation'] == dict(generation, max_new_tokens=expected_tokens)
    else:
        assert 'region coordinates' in content
        if missing: assert 'mitoses' in content


@pytest.mark.parametrize('config', CONFIGS)
@pytest.mark.parametrize('greedy', [False,True])
def test_executor_sampling_and_thinking_flags(config, greedy, monkeypatch):
    backend, calls, messages, _ = generative_backend(config, monkeypatch, '{"answer":"A"}')
    backend._text([{'role':'user','content':'question'}], 512, 'final' if greedy else 'predict', greedy)
    passed = {k:v for k,v in calls[0].items() if k not in {'input_ids','image_grid_thw'}}
    expected = {'max_new_tokens':512}
    if greedy: expected.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    assert passed == expected
    assert messages[0][1]['enable_thinking'] is False


def test_paper_context_budget_includes_retry_and_compresses_without_dropping():
    backend = object.__new__(PaperModelBackend)
    backend.config = load_config(CONFIGS[1])
    backend.generation = backend.config['generation']
    backend.context_limit = 6000
    envelopes, summaries = [], []
    class Tokenizer:
        def encode(self, text): return list(text)
        def apply_chat_template(self, messages, **kwargs):
            envelopes.append(messages)
            return ''.join(m['content'] for m in messages)
    backend.tokenizer = Tokenizer()
    def summarize(messages, tokens, stage, greedy=False):
        summaries.append(messages)
        return 'compressed observations'
    backend._text = summarize
    regions = [Region(str(i), '/unused', i*40, 0, 40, 40, 10, 10, 'physical',5,.5) for i in range(2)]
    findings = [{'region':r.to_dict(), 'description':'x'*1800} for r in regions]
    result = backend._evidence_for(findings, 'system', lambda d: 'p'*500+d, 2048, 'Which?')
    assert result == 'compressed observations'
    assert len(summaries) == 1
    assert all(r.metadata() in summaries[0][1]['content'] for r in regions)
    assert all(e[-1]['content'].startswith('Return one valid JSON') for e in envelopes)


def test_observation_caches_isolate_protocol_even_with_same_directory(tmp_path):
    from data_processing.preprocess import prepare_observations
    region = Region('one','/unused',0,0,40,40,10,10,'physical',5,.5)
    class Backend:
        records = []
        def set_case(self, case): pass
        def describe(self, region):
            self.records.append({'stage':'describe'})
            return self.description
        def encode_regions(self, regions): return np.array([[1.,0.]])
    backend = Backend()
    old, paper = map(load_config, CONFIGS)
    # Deliberately equalize budgets: isolation must also depend on prompt/protocol identity.
    paper['generation'] = copy.deepcopy(old['generation'])
    backend.description = 'legacy evidence'
    assert prepare_observations(tmp_path, [region], backend, old)[0]['one'] == 'legacy evidence'
    backend.description = 'paper evidence'
    assert prepare_observations(tmp_path, [region], backend, paper)[0]['one'] == 'paper evidence'
    assert prepare_observations(tmp_path, [region], backend, old)[0]['one'] == 'legacy evidence'


@pytest.mark.parametrize('config', CONFIGS)
def test_trident_uses_local_interpreter_for_both_protocols(config, tmp_path, monkeypatch):
    import data_processing.preprocess as preprocessing
    from data_processing.common import resolve
    cfg = load_config(config)
    cfg['runtime']['processed_root'] = str(tmp_path / 'cache')
    image = tmp_path / 'slide.png'
    Image.new('RGB', (16,16)).save(image)
    commands = []
    def run(command, check):
        commands.append(command)
        assert check
        request = read_json(command[-1])
        assert request['workers'] == 1
        atomic_json(request['output'], {'coords':[[0,0]]})
    monkeypatch.setattr(preprocessing.subprocess, 'run', run)
    directory, regions = preprocessing.prepare_regions(
        {'image_path':str(image), 'input_kind':'wsi', 'assumed_mpp':.5, 'dataset':'bcnb'}, cfg, 'source')
    assert len(regions) == 1 and (directory/'regions.json').exists()
    assert commands == [[str(resolve(cfg['preprocessing']['trident_python'])),
                         str(resolve('scripts/trident_coords.py')), '--request', str(directory/'request.json')]]


@pytest.mark.parametrize('config,expected_calls,expected_status', [
    (CONFIGS[0],2,'ok'), (CONFIGS[1],1,'invalid'), ('configs/reproduce_v3.yaml',1,'invalid')])
def test_runner_resume_and_cross_protocol_rejection(config, expected_calls, expected_status, tmp_path, monkeypatch):
    import torch
    import scripts.run_inference as runner
    from data_processing.datasets import make_sample
    for name in ['PATHAGENT_EXPECT_CODE_HASH','PATHAGENT_EXPECT_GPU_NAME','PATHAGENT_EXPECT_GPU_MEMORY_BYTES','PATHAGENT_GPU_FLEET_LOCK']:
        monkeypatch.delenv(name, raising=False)
    cfg = load_config(config)
    cfg['runtime']['processed_root'] = str(tmp_path / 'cache')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    image = tmp_path / 'roi.png'
    Image.new('RGB',(16,16)).save(image)
    row = make_sample('bcnb','test','one','image',str(image),'Which?',{'A':'yes','B':'no'},'A')
    row.update(input_kind='roi', assumed_mpp=None)
    manifest = tmp_path / 'manifest.json'
    write_manifest(manifest,[row],dataset='bcnb',purpose='full',images_verified=True)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys,'argv',['pathagent.py','--config',str(config_path),'--manifest',str(manifest),'--output',str(output)])
    monkeypatch.setattr(runner,'check_assets',lambda c:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'fake CPU test')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=48*1024**3))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(runner,'seed_everything',lambda seed:None)
    class Backend:
        load_seconds = 0
        def __init__(self, config): self.records, self.resolved_generation = [], {}
        def set_case(self, case): pass
        def close(self): pass
    monkeypatch.setattr(runner,'backend_class',lambda config:Backend)
    monkeypatch.setattr(runner,'prepare_observations',lambda *args:({},{}))
    calls = []
    def agent(*args):
        calls.append(1)
        if len(calls)==1: raise ModelProtocolError('invalid final')
        return {'status':'ok','pred_answer':'A'}
    monkeypatch.setattr(runner,'run_agent',agent)
    runner.main()
    runner.main()
    assert len(calls) == expected_calls
    result = read_json(next((output/'results').glob('*.json')))
    assert result['status'] == expected_status
    assert read_json(output/'prompt_manifest.json')['protocol'] == protocol_name(cfg)
    assert read_json(output/'environment.json')['executable'] == sys.executable
    assert read_json(output/'image_inventory.json')[row['image_id']]['observation_cache'].startswith(str(tmp_path/'cache'))
    # Reusing an output directory for the other protocol must fail before model loading.
    other = load_config(CONFIGS[1] if config==CONFIGS[0] else CONFIGS[0])
    config_path.write_text(yaml.safe_dump(other))
    with pytest.raises(ValueError, match='Resume refused'): runner.main()


@pytest.mark.parametrize('module_name', ['scripts.submit_runs','slurm.submit_benchmark','scripts.preflight'])
def test_cli_config_is_passed_to_loader(module_name, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    calls=[]
    class Selected(Exception): pass
    def load(path):
        calls.append(path)
        raise Selected
    monkeypatch.setattr(module,'load_config',load)
    args = ['tool','--config',CONFIGS[1]]
    if module_name=='scripts.submit_runs': args += ['smoke','--dataset','bcnb']
    if module_name=='slurm.submit_benchmark': args += ['bcnb','--skip-smoke']
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(Selected): module.main()
    assert calls == [CONFIGS[1]]


def test_slurm_uses_active_python_and_selected_frozen_config(tmp_path):
    from scripts.submit_runs import script_text
    config = tmp_path / 'paper config.yaml'
    text = script_text(tmp_path/'manifest',tmp_path/'output',config,'sourcehash',array=True)
    command = shlex.split(text.splitlines()[-1])
    assert command[:3] == ['exec',sys.executable,'pathagent.py']
    assert command[command.index('--config')+1] == str(config)
    assert '/localtmp/' not in text and 'activate.sh' not in text


def test_smoke_verification_uses_same_identity_as_runner(tmp_path, monkeypatch):
    import scripts.submit_runs as submitter
    cfg = load_config(CONFIGS[1])
    samples = [{'sample_id':'one'}]
    manifest = {'samples_hash':digest(samples),'samples':samples}
    identity = run_identity(cfg,'source',manifest['samples_hash'])
    run_hash = digest(identity)
    atomic_json(tmp_path/'run_lock.json',dict(identity,run_hash=run_hash))
    atomic_json(tmp_path/'profile.json',{'passed':True,'real_gpu':True,'smoke':True,'dataset':'bcnb',
        'slurm_job_id':'test','code_hash':'source','config_hash':digest(cfg),
        'manifest_hash':manifest['samples_hash'],'run_hash':run_hash})
    atomic_json(tmp_path/'results'/(digest('one')+'.json'),{'status':'ok','run_hash':run_hash})
    monkeypatch.setattr(submitter,'read_manifest',lambda path:manifest)
    assert submitter.verify_smoke(tmp_path,'bcnb',cfg,'source')['passed']
    with pytest.raises(RuntimeError,match='stale'):
        submitter.verify_smoke(tmp_path,'bcnb',load_config(CONFIGS[0]),'source')
