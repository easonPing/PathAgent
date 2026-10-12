"""CPU-only contracts for isolated prompt variants and their provenance."""
import ast
import hashlib
import json
from pathlib import Path
from string import Formatter
import sys
from types import ModuleType

import pytest

from slurm import rule_prompts as rules


ROOT = Path(__file__).resolve().parents[1]
PAPER = (ROOT / 'models/paper_prompts.py').read_text(encoding='utf-8')
CONTEXT = (ROOT / 'models/context_prompts.py').read_text(encoding='utf-8')
EXPECTED_CHANGED = {
    'O_ONLY': ({'PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'QUESTION_PROMPT', 'PREDICT_SYSTEM'}, set()),
    'D_ONLY': ({'PREDICT_SYSTEM'}, set()),
    'R_ONLY': ({'REFLECT_SYSTEM', 'REFLECT_USER'}, set()),
    'M_ONLY': ({'MISSING_SYSTEM'}, {'ZOOM_MISSING_USER'}),
    'F_ONLY': ({'FINAL_SYSTEM', 'SUMMARY_SYSTEM'}, set()),
}


def values(source):
    return rules._literal_strings(source)


def fields(value):
    return {name for _, name, _, _ in Formatter().parse(value) if name is not None}


def run_manifests(sources, monkeypatch):
    paper = ModuleType('models.paper_prompts')
    exec(compile(sources['models/paper_prompts.py'], 'paper_prompts.py', 'exec'), paper.__dict__)
    protocols = ModuleType('models.protocols')
    protocols.CONTEXT_PROTOCOL = 'paper-prompts-context-restored-v3'
    monkeypatch.setitem(sys.modules, 'models.paper_prompts', paper)
    monkeypatch.setitem(sys.modules, 'models.protocols', protocols)
    context = ModuleType('models.context_prompts')
    exec(compile(sources['models/context_prompts.py'], 'context_prompts.py', 'exec'), context.__dict__)
    return paper.prompt_manifest(), context.prompt_manifest()


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_only_selected_package_templates_change(variant):
    rendered = rules.render_variant_sources(PAPER, CONTEXT, variant)
    originals = (values(PAPER), values(CONTEXT))
    actual = tuple(values(rendered[path]) for path in
                   ('models/paper_prompts.py', 'models/context_prompts.py'))
    expected = EXPECTED_CHANGED.get(variant)
    if expected is None:
        expected = tuple(set.union(*(item[i] for item in EXPECTED_CHANGED.values()))
                         for i in (0, 1))
    for before, after, names in zip(originals, actual, expected):
        assert {name for name in before if before[name] != after[name]} == names
        assert after['PROTOCOL'] == before['PROTOCOL'] if 'PROTOCOL' in before else True
        for name, value in before.items():
            if name not in names:
                assert after[name] == value
    # Rendering is deterministic, and never writes the base prompt files.
    assert rendered == rules.build_variant_sources(ROOT, variant)
    assert (ROOT / 'models/paper_prompts.py').read_text(encoding='utf-8') == PAPER
    assert (ROOT / 'models/context_prompts.py').read_text(encoding='utf-8') == CONTEXT


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_format_fields_and_original_model_output_keys_are_preserved(variant):
    rendered = rules.render_variant_sources(PAPER, CONTEXT, variant)
    for source, target in ((PAPER, rendered['models/paper_prompts.py']),
                           (CONTEXT, rendered['models/context_prompts.py'])):
        before, after = values(source), values(target)
        for name, original in before.items():
            if name.endswith('_USER') or name in ('QUESTION_PROMPT',):
                assert fields(after[name]) == fields(original)
                supplied = {field: 'value' for field in fields(original)}
                assert isinstance(after[name].format(**supplied), str)
    paper = values(rendered['models/paper_prompts.py'])
    for name, required in {
        'PREDICT_SYSTEM': ('"answer"', '"thinking_steps"'),
        'MISSING_SYSTEM': ('"missing_info"', '"zoom_recommendation"',
                           '"recommended_zoom_level"', '"zoom_reason"'),
    }.items():
        assert all(key in paper[name] for key in required)
    assert 'keys "answer" and "explanation"' in paper['FINAL_USER']


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_changed_manifest_attribution_retains_original_source_and_hash(variant, monkeypatch):
    base_sources = {'models/paper_prompts.py': PAPER, 'models/context_prompts.py': CONTEXT}
    baselines = run_manifests(base_sources, monkeypatch)
    rendered = rules.render_variant_sources(PAPER, CONTEXT, variant)
    manifests = run_manifests(rendered, monkeypatch)
    for before, after in zip(baselines, manifests):
        assert after['protocol'] == before['protocol']
        assert after['variant'] == variant
        assert after['rules'] == list(rules.selected_packages(variant))
        assert after['rule_version'] == 'rules-v0'
        for name, item in after['templates'].items():
            original = before['templates'][name]
            assert item['sha256'] == hashlib.sha256(item['text'].encode()).hexdigest()
            if item['text'] == original['text']:
                assert item == original
            else:
                assert item['published'] is False
                assert item['base_hash'] == original['sha256']
                assert item['base_source'] == original['source']
                assert item['base_published'] == original['published']
                assert item['source'] == 'local rules-v0 prompt variant ' + variant


def test_all_package_order_and_capsule_routing():
    paper = values(rules.build_variant_sources(ROOT, 'ALL')['models/paper_prompts.py'])
    for name, packages in {'PREDICT_SYSTEM': ('O', 'D')}.items():
        offsets = [paper[name].index('[rules-v0:' + p + ']') for p in packages]
        assert offsets == sorted(offsets)
    assert 'Apply only the capsule matching the question and options.' in paper['PREDICT_SYSTEM']
    for capsule in ('Lung', 'Endometrium', 'Breast Tumor', 'Breast Nottingham',
                    'Melanoma', 'Receptors/Molecular'):
        assert '[' + capsule + ']' in paper['PREDICT_SYSTEM']


def test_reflect_system_and_user_declare_same_compatible_extension():
    paper = values(rules.build_variant_sources(ROOT, 'R_ONLY')['models/paper_prompts.py'])
    for name in ('REFLECT_SYSTEM', 'REFLECT_USER'):
        assert rules.REFLECT_SCHEMA in paper[name]
        assert '"reason"' in paper[name] and '"failed_checks"' in paper[name]
        assert 'a JSON object: {"sufficient"' not in paper[name]
        assert 'a JSON object: {{"sufficient"' not in paper[name]


def test_extra_reflection_keys_reach_missing_and_final_without_model_loading(monkeypatch):
    # Exercise the actual parser, validator, reflection forwarding and history
    # builder with scripted strings. No model constructor or torch import runs.
    from models.context_inference import ContextModelBackend
    import models.paper_inference as inference
    rendered = rules.build_variant_sources(ROOT, 'R_ONLY')
    paper = values(rendered['models/paper_prompts.py'])
    for name in ('REFLECT_SYSTEM', 'REFLECT_USER'):
        monkeypatch.setattr(inference, name, paper[name])
    backend = object.__new__(ContextModelBackend)
    backend.generation = dict(step_a_tokens=1024, step_b_tokens=512,
                              step_c_tokens=512, final_tokens=2048, retries=1)
    backend._evidence_for = lambda *args, **kwargs: 'Region=r1; conflicting observations'
    reflection = {'sufficient': 'No', 'reason': 'r1 contains conflicting observations',
                  'failed_checks': ['same_region_consistency']}
    responses = iter(map(json.dumps, [
        {'answer': 'A', 'thinking_steps': 'tentative'}, reflection,
        {'missing_info': 'glandular structure', 'zoom_recommendation': 'Yes',
         'recommended_zoom_level': 2, 'zoom_reason': 'retain local context'},
        {'answer': 'A', 'explanation': 'uncertain estimate'},
    ]))
    calls = []
    def text(messages, tokens, stage, greedy=False):
        calls.append((stage, messages))
        return next(responses)
    backend._text = text
    sample = {'question': 'Which?', 'choices': {'A': 'one', 'B': 'two'}}
    decision = backend.evaluate([], sample, 'relative', [2, 4, 8])
    assert decision['reason'] == reflection['reason']
    assert decision['failed_checks'] == reflection['failed_checks']
    assert decision['zoom_level'] == 2
    assert json.dumps(reflection) in calls[-1][1][1]['content']
    backend.conclude([], [{'iteration': 1, 'action': 'zoom', 'decision': decision}], sample)
    assert reflection['reason'] in calls[-1][1][1]['content']
    assert 'same_region_consistency' in calls[-1][1][1]['content']


def test_magnification_rule_respects_one_zoom_and_scale_semantics():
    rendered = rules.build_variant_sources(ROOT, 'M_ONLY')
    paper, context = (values(rendered[path]) for path in
                       ('models/paper_prompts.py', 'models/context_prompts.py'))
    prompt = paper['MISSING_SYSTEM']
    assert 'one zoom available before Final' in prompt
    assert 'do not require a sequence of magnifications' in prompt
    assert 'lowest allowed scale expected to be sufficient' in prompt
    assert '2/4/8 are crop factors, not physical magnification' in prompt
    assert 'JSON integer from the supplied allowed scales' in prompt
    assert 'Zoom cannot obtain unprovided IHC' in prompt
    assert 'whole-slide absence' in context['ZOOM_MISSING_USER']
    assert paper['GENERIC_PROMPT'] == values(PAPER)['GENERIC_PROMPT']


def test_compile_is_pure_rejects_unknown_variant_and_double_application():
    paper, context = values(PAPER), values(CONTEXT)
    before = (dict(paper), dict(context))
    rules.compile_variant(paper, context, 'ALL')
    assert (paper, context) == before
    with pytest.raises(ValueError, match='Unknown'):
        rules.render_variant_sources(PAPER, CONTEXT, 'BASE')
    rendered = rules.render_variant_sources(PAPER, CONTEXT, 'O_ONLY')
    with pytest.raises(ValueError, match='unmodified base'):
        rules.render_variant_sources(rendered['models/paper_prompts.py'],
                                     rendered['models/context_prompts.py'], 'ALL')


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_snapshot_changes_only_string_assignments_and_manifest_overlay(variant):
    rendered = rules.build_variant_sources(ROOT, variant)
    for source, output in ((PAPER, rendered['models/paper_prompts.py']),
                           (CONTEXT, rendered['models/context_prompts.py'])):
        base_ast = ast.parse(source)
        changed_ast = ast.parse(output.split(rules._METADATA_MARKER)[0])
        for tree in (base_ast, changed_ast):
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        node.value.value = 'string placeholder'
        assert ast.dump(base_ast) == ast.dump(changed_ast)
