"""CPU-only contracts for the short A/B/C/D experiment prompt compiler."""
import ast
import hashlib
import json
from pathlib import Path
from string import Formatter
import sys
from types import ModuleType

import pytest

from slurm import rule_prompts as v0
from slurm import rule_prompts_v1 as rules


ROOT = Path(__file__).resolve().parents[1]
PAPER = (ROOT / 'models/paper_prompts.py').read_text(encoding='utf-8')
CONTEXT = (ROOT / 'models/context_prompts.py').read_text(encoding='utf-8')
ARMS = (rules.CONTROL,) + rules.VARIANTS
FOUR = {'PREDICT_SYSTEM', 'REFLECT_SYSTEM', 'MISSING_SYSTEM', 'FINAL_SYSTEM'}
CHANGED = {
    'BASE_COMPAT': set(), 'A_HE_PRED': FOUR, 'B_CONFLICT': FOUR,
    'C_ZOOM_LOOP': set(), 'D_SCALE': {'REFLECT_SYSTEM', 'MISSING_SYSTEM'},
    'ALL4': FOUR,
}
PERCEPTOR = {'GENERIC_PROMPT', 'QUESTION_PROMPT', 'PERCEPTOR_SYSTEM'}


def vals(source):
    return v0._literal_strings(source)


def manifests(sources, monkeypatch):
    paper = ModuleType('models.paper_prompts')
    exec(compile(sources['models/paper_prompts.py'], 'paper.py', 'exec'), paper.__dict__)
    protocols = ModuleType('models.protocols')
    protocols.CONTEXT_PROTOCOL = 'paper-prompts-context-restored-v3'
    monkeypatch.setitem(sys.modules, 'models.paper_prompts', paper)
    monkeypatch.setitem(sys.modules, 'models.protocols', protocols)
    context = ModuleType('models.context_prompts')
    exec(compile(sources['models/context_prompts.py'], 'context.py', 'exec'), context.__dict__)
    return paper.prompt_manifest(), context.prompt_manifest()


def test_public_interfaces_and_toggle_are_exact():
    assert rules.CONTROL == 'BASE_COMPAT'
    assert rules.VARIANTS == ('A_HE_PRED', 'B_CONFLICT', 'C_ZOOM_LOOP', 'D_SCALE', 'ALL4')
    expected = {'BASE_COMPAT': (), 'A_HE_PRED': ('A',), 'B_CONFLICT': ('B',),
                'C_ZOOM_LOOP': (), 'D_SCALE': ('D',), 'ALL4': ('A', 'B', 'D')}
    for arm in ARMS:
        assert rules.selected_packages(arm) == rules.packages(arm) == expected[arm]
        assert rules.zoom_enabled(arm) is (arm in {'C_ZOOM_LOOP', 'ALL4'})
        assert rules.ZOOM_ENABLED[arm] is rules.zoom_enabled(arm)


@pytest.mark.parametrize('arm', ARMS)
def test_exact_stage_matrix_and_perceptor_user_wrapper_identity(arm):
    rendered = rules.build_variant_sources(ROOT, arm)
    assert set(rendered) == {'models/paper_prompts.py', 'models/context_prompts.py'}
    paper, context = vals(rendered['models/paper_prompts.py']), vals(rendered['models/context_prompts.py'])
    before = vals(PAPER)
    assert {name for name in before if before[name] != paper[name]} == CHANGED[arm]
    assert context == vals(CONTEXT)
    assert all(paper[name] == before[name] for name in PERCEPTOR)
    assert all(paper[name] == value for name, value in before.items()
               if name.endswith('_USER') or name in ('SUMMARY_SYSTEM', 'PROTOCOL'))
    if arm in ('BASE_COMPAT', 'C_ZOOM_LOOP'):
        assert paper == before
    assert rendered == rules.render_variant_sources(PAPER, CONTEXT, arm)


@pytest.mark.parametrize('arm', ARMS)
def test_no_experiment_labels_leak_into_actual_prompt_messages(arm):
    rendered = rules.build_variant_sources(ROOT, arm)
    for source in rendered.values():
        for name, value in vals(source).items():
            if name.endswith(('_SYSTEM', '_USER', '_PROMPT')):
                assert 'rules-v0' not in value and 'rules-v1' not in value
                assert '[rules-' not in value
                assert not any(label in value for label in ARMS)


@pytest.mark.parametrize('arm', ARMS)
def test_manifest_keeps_unchanged_template_items_and_changed_base_provenance(arm, monkeypatch):
    baseline = manifests({'models/paper_prompts.py': PAPER,
                          'models/context_prompts.py': CONTEXT}, monkeypatch)
    result = manifests(rules.build_variant_sources(ROOT, arm), monkeypatch)
    for base, actual in zip(baseline, result):
        assert actual['protocol'] == base['protocol']
        assert actual['notes'] == base['notes']
        assert actual['variant'] == arm
        assert actual['rules'] == list(rules.selected_packages(arm))
        assert actual['post_zoom_reflect'] is rules.zoom_enabled(arm)
        assert actual['rule_version'] == 'rules-v1'
        assert set(actual['templates']) == set(base['templates'])
        for name, item in actual['templates'].items():
            old = base['templates'][name]
            assert item['sha256'] == hashlib.sha256(item['text'].encode()).hexdigest()
            if name in CHANGED[arm]:
                assert item['text'] != old['text']
                assert item['published'] is False
                assert item['source'] == 'local rules-v1 prompt variant ' + arm
                assert item['base_hash'] == old['sha256']
                assert item['base_source'] == old['source']
                assert item['base_published'] == old['published']
            else:
                # Check serialized bytes as well as semantic equality: no
                # extra provenance keys/reordered items for unchanged prompts.
                assert json.dumps(item, ensure_ascii=False) == json.dumps(old, ensure_ascii=False)


def test_a_keeps_original_non_he_evidence_requirements_verbatim():
    paper = vals(rules.build_variant_sources(ROOT, 'A_HE_PRED')['models/paper_prompts.py'])
    assert ('for other questions, judge whether the current patch descriptions '
            'are sufficient to confidently support the answer.') in paper['REFLECT_SYSTEM']
    assert ('For other questions: ' + rules.BASE_FINAL_REQUIREMENT) in paper['FINAL_SYSTEM']
    assert paper['FINAL_SYSTEM'].count(rules.BASE_FINAL_REQUIREMENT) == 1
    assert rules.A_REFLECT in paper['REFLECT_SYSTEM']
    assert 'without requiring an assay' in paper['REFLECT_SYSTEM']
    assert 'indirect morphological estimate' in paper['FINAL_SYSTEM']


def test_all4_composes_a_then_b_then_d_without_changing_wrappers():
    p = vals(rules.build_variant_sources(ROOT, 'ALL4')['models/paper_prompts.py'])
    for name, chunks in {
        'PREDICT_SYSTEM': (rules.A_PREDICT, rules.B_PREDICT),
        'REFLECT_SYSTEM': (rules.A_REFLECT, rules.B_REFLECT, rules.D_REFLECT),
        'MISSING_SYSTEM': (rules.A_MISSING, rules.B_MISSING, rules.D_MISSING),
        'FINAL_SYSTEM': (rules.A_FINAL, rules.B_FINAL),
    }.items():
        offsets = [p[name].index(chunk) for chunk in chunks]
        assert offsets == sorted(offsets)
        assert all(p[name].count(chunk) == 1 for chunk in chunks)


@pytest.mark.parametrize('arm', ARMS)
def test_reflect_stays_sufficient_only_and_all_format_fields_are_preserved(arm):
    source = rules.build_variant_sources(ROOT, arm)
    p = vals(source['models/paper_prompts.py'])
    assert p['REFLECT_SYSTEM'].count('Output ONLY a JSON object: {"sufficient": "Yes" or "No"}.') == 1
    assert p['REFLECT_USER'] == vals(PAPER)['REFLECT_USER']
    assert 'failed_checks' not in p['REFLECT_SYSTEM']
    assert '"reason"' not in p['REFLECT_SYSTEM']
    for base, output in ((PAPER, source['models/paper_prompts.py']),
                         (CONTEXT, source['models/context_prompts.py'])):
        before, after = vals(base), vals(output)
        for name, text in before.items():
            if name.endswith('_USER') or name == 'QUESTION_PROMPT':
                fields = {f for _, f, _, _ in Formatter().parse(text) if f is not None}
                actual = {f for _, f, _, _ in Formatter().parse(after[name]) if f is not None}
                assert actual == fields
                after[name].format(**{f: 'value' for f in fields})


def test_b_conflict_logging_uses_existing_fields_and_d_has_fixed_scale_map():
    b = vals(rules.build_variant_sources(ROOT, 'B_CONFLICT')['models/paper_prompts.py'])
    assert 'same feature' in b['PREDICT_SYSTEM']
    assert 'thinking_steps' in b['PREDICT_SYSTEM']
    assert 'An omission or a difference between regions is not itself a contradiction.' in b['REFLECT_SYSTEM']
    assert 'neutral visual feature' in b['MISSING_SYSTEM']
    assert 'Region ID in zoom_reason' in b['MISSING_SYSTEM']
    d = vals(rules.build_variant_sources(ROOT, 'D_SCALE')['models/paper_prompts.py'])
    for clause in ('5x for distribution, localization and representativeness',
                   '10x for glands, cell nests and large architecture',
                   '20x for cell cohesion, arrangement and nuclei',
                   '40x for fine cellular features and mitoses'):
        assert clause in d['REFLECT_SYSTEM']
    assert '(20x if needed)' in d['MISSING_SYSTEM']
    assert '(40x if needed)' in d['MISSING_SYSTEM']
    assert 'zoom_recommendation to No to explore another 5x region' in d['MISSING_SYSTEM']
    assert 'neutral visual feature' not in d['MISSING_SYSTEM']
    assert d['FINAL_SYSTEM'] == vals(PAPER)['FINAL_SYSTEM']


@pytest.mark.parametrize('arm', ARMS)
def test_ast_changes_only_string_assignments_and_manifest_metadata(arm):
    rendered = rules.build_variant_sources(ROOT, arm)
    for base, output in ((PAPER, rendered['models/paper_prompts.py']),
                         (CONTEXT, rendered['models/context_prompts.py'])):
        trees = [ast.parse(base), ast.parse(output.split(rules._METADATA_MARKER, 1)[0])]
        for tree in trees:
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    node.value.value = 'normalized string'
        assert ast.dump(trees[0]) == ast.dump(trees[1])


def test_compile_is_pure_rejects_unknown_and_already_compiled_sources():
    before_paper, before_context = vals(PAPER), vals(CONTEXT)
    original = (dict(before_paper), dict(before_context))
    rules.compile_variant(before_paper, before_context, 'ALL4')
    assert (before_paper, before_context) == original
    for fn in (rules.selected_packages, rules.zoom_enabled):
        with pytest.raises(ValueError, match='Unknown rules-v1'):
            fn('A_ONLY')
    for renderer, variant in ((rules, 'BASE_COMPAT'), (v0, 'O_ONLY')):
        modified = renderer.build_variant_sources(ROOT, variant)
        with pytest.raises(ValueError, match='unmodified base'):
            rules.render_variant_sources(modified['models/paper_prompts.py'],
                                         modified['models/context_prompts.py'], 'ALL4')


def test_approved_additions_are_short_and_contain_no_extra_control_actions():
    for text in (rules.A_PREDICT, rules.A_REFLECT, rules.A_MISSING, rules.A_FINAL,
                 rules.B_PREDICT, rules.B_REFLECT, rules.B_MISSING, rules.B_FINAL,
                 rules.D_REFLECT, rules.D_MISSING):
        assert len(text.split()) <= 75
        assert text.count('.') <= 2
        assert 'failed_checks' not in text
        assert 'STOP' not in text and 'ABSTAIN' not in text
