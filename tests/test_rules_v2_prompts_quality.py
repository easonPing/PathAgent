"""Contracts that protect the archived baseline and conservative ROI filtering."""
import copy
import hashlib
from pathlib import Path
from string import Formatter

import numpy as np
import pytest
from PIL import Image

from data_processing.regions import Region
from data_processing.roi_quality import quality_for_region
from models import context_prompts, paper_prompts
from models import rules_v2_prompts as rules


def config(variant):
    return {'protocol': rules.PROTOCOL, 'rules_v2': {'variant': variant}}


def baseline():
    values = {n: v for n, v in vars(paper_prompts).items()
              if n.endswith(('_SYSTEM', '_USER', '_PROMPT')) and isinstance(v, str)}
    for name in ('PERCEPTOR_USER', 'ZOOM_MISSING_USER', 'MISSING_USER'):
        values[name] = getattr(context_prompts, name)
    return values


def test_pure_compilation_preserves_baseline_sources_and_config():
    before = baseline()
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'models' / name for name in ('paper_prompts.py', 'context_prompts.py')]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    for variant in rules.VARIANTS:
        conf = config(variant)
        original = copy.deepcopy(conf)
        first = rules.templates_for_config(conf)
        assert first == rules.templates_for_config(conf)
        assert conf == original
    assert baseline() == before
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_single_rules_do_not_change_unrelated_upstream_templates():
    before = baseline()
    for variant in ('R1_EVIDENCE', 'R4_VISUAL_MISSING', 'R5_TARGET_REGION'):
        actual = rules.templates_for_config(config(variant))
        for name in ('PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'QUESTION_PROMPT', 'SUMMARY_SYSTEM'):
            assert actual[name] == before[name]
    r3 = rules.templates_for_config(config('R3_ROI_VALIDITY'))
    for name in ('PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'SUMMARY_SYSTEM'):
        assert r3[name] == before[name]
    assert 'tissue_status' in r3['QUESTION_PROMPT']
    r2 = rules.templates_for_config(config('R2_DIFFERENTIAL'))
    all5 = rules.templates_for_config(config('ALL5'))
    assert all5['GENERIC_PROMPT'] == r2['GENERIC_PROMPT']
    assert all5['PERCEPTOR_SYSTEM'] == r2['PERCEPTOR_SYSTEM']
    assert 'tissue_status' not in r2['QUESTION_PROMPT']
    assert 'tissue_status' in all5['QUESTION_PROMPT']


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_effective_manifest_hashes_and_no_experiment_labels(variant):
    manifest = rules.prompt_manifest(config(variant))
    actual = rules.templates_for_config(config(variant))
    assert manifest['protocol'] == rules.PROTOCOL
    assert manifest['post_zoom_reflect'] is True
    assert manifest['zoom_selected_children'] == 5
    assert manifest['rules'] == [f for f in rules.RULES if f in rules.flags_for_variant(variant)]
    for name, text in actual.items():
        item = manifest['templates'][name]
        assert item['text'] == text
        assert item['sha256'] == hashlib.sha256(text.encode()).hexdigest()
        assert item['sha256'] == manifest['effective_template_hashes'][name]
        assert not any(label in text for label in rules.VARIANTS)
        if name not in baseline() or text != baseline()[name]:
            assert item['published'] is False
            assert item['base_hash'] == manifest['original_template_hashes'][name]


def test_r3_question_only_format_contract_and_retry_are_manifested():
    for variant in rules.VARIANTS:
        compiled = rules.templates_for_config(config(variant))
        manifest = rules.prompt_manifest(config(variant))
        for name in ('QUESTION_OUTPUT_SYSTEM', 'QUESTION_OUTPUT_USER', 'QUESTION_FORMAT_RETRY_USER'):
            selected = variant in ('R3_ROI_VALIDITY', 'ALL5')
            assert (name in compiled) is selected
            if selected:
                assert manifest['templates'][name]['sha256'] == hashlib.sha256(compiled[name].encode()).hexdigest()
                assert manifest['templates'][name]['base_hash'] is None
        if variant in ('R3_ROI_VALIDITY', 'ALL5'):
            retry = compiled['QUESTION_FORMAT_RETRY_USER'].format(previous_response='malformed previous text')
            assert '"tissue_status"' in retry and '"findings"' in retry
            assert 'malformed previous text' in retry


def test_complete_schema_union_and_user_wrappers_are_consistent():
    flags = rules.flags_for_variant('ALL5')
    assert rules.predict_keys(flags) == ('answer', 'thinking_steps', 'required_features',
                                        'alternative_answer', 'discriminating_feature', 'conflicts')
    assert rules.reflect_keys(flags) == ('sufficient', 'checks', 'unresolved_conflicts', 'target_observed')
    assert rules.missing_keys(flags) == ('missing_info', 'zoom_recommendation', 'recommended_zoom_level',
                                        'zoom_reason', 'evidence_type', 'target_region')
    effective = rules.templates_for_config(config('ALL5'))
    for name in ('PREDICT_SYSTEM', 'REFLECT_SYSTEM', 'MISSING_SYSTEM'):
        assert effective[name].count('Output ONLY a JSON object') == 1
    assert 'all required fields' in effective['REFLECT_USER']
    assert '"sufficient": "Yes" or "No"' not in effective['REFLECT_USER']
    assert 'all required fields' in effective['PREDICT_USER']
    for name, original in baseline().items():
        if name.endswith('_USER') or name == 'QUESTION_PROMPT':
            fields = {field for _, field, _, _ in Formatter().parse(original) if field}
            actual_fields = {field for _, field, _, _ in Formatter().parse(effective[name]) if field}
            assert actual_fields == fields
            effective[name].format(**{field: 'value' for field in fields})


def test_scale_and_other_question_requirements_remain_explicit():
    r1 = rules.templates_for_config(config('R1_EVIDENCE'))
    assert ('judge whether the current patch descriptions are sufficient to '
            'confidently support the answer.') in r1['REFLECT_SYSTEM']
    assert 'indirect morphological estimate' in r1['REFLECT_SYSTEM']
    assert 'metadata as the scale authority' in r1['REFLECT_SYSTEM']
    assert rules.REQUIRED_SCALES == {'overview': 5, 'architecture': 10,
                                     'cellular': 20, 'fine_detail': 40}
    r2 = rules.templates_for_config(config('R2_DIFFERENTIAL'))
    assert 'For other questions: ' + rules.BASE_FINAL_REQUIREMENT in r2['FINAL_SYSTEM']
    assert 'neutral observable feature' in r2['MISSING_SYSTEM']
    assert 'Omissions and differences between regions' in r2['REFLECT_SYSTEM']


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_compact_scale_checks_are_isolated_and_keep_required_feature_contract(variant):
    effective = rules.templates_for_config(config(variant))
    clauses = {
        'PREDICT_SYSTEM': ('one to four required_features groups',
                           'at most eight words',
                           'every feature used in concise thinking_steps',
                           'Assign mitoses, mitotic figures, mitotic activity and mitotic counts to fine_detail'),
        'REFLECT_SYSTEM': ('copy its exact feature string into one check',
                           'one anchor region_id',
                           'compact JSON', 'coverage using all available Regions'),
    }
    selected = variant in ('R1_EVIDENCE', 'ALL5')
    for stage, values in clauses.items():
        for clause in values:
            assert (clause in effective[stage]) is selected
    # The change reduces payload through the prompt, not through dropped
    # checks, altered schemas or expanded generation budgets.
    if selected:
        flags = rules.flags_for_variant(variant)
        schema = rules.schemas_for_flags(flags)
        assert schema['predict']['required_features'] == [
            {'feature': 'string', 'evidence_type': '|'.join(rules.REQUIRED_SCALES)}]
        assert tuple(schema['reflect']['checks'][0]) == (
            'feature', 'evidence_type', 'region_id', 'assessable', 'coverage_sufficient')
        assert 'below its minimum magnification' in effective['REFLECT_SYSTEM']


@pytest.mark.parametrize('variant', rules.VARIANTS)
def test_distinct_listed_differential_choices_are_isolated_to_r2(variant):
    effective = rules.templates_for_config(config(variant))
    assert ('Compare two distinct listed choices' in effective['PREDICT_SYSTEM']) is (
        variant in ('R2_DIFFERENTIAL', 'ALL5'))


@pytest.mark.parametrize('variant', ('R2_DIFFERENTIAL', 'ALL5'))
def test_conflict_contract_requests_a_real_pair_without_forcing_conflicts(variant):
    manifest = rules.prompt_manifest(config(variant))
    claims = manifest['schemas']['predict']['conflicts'][0]['claims']
    assert claims == ['first opposing claim (string)', 'second opposing claim (string)']
    system = manifest['templates']['PREDICT_SYSTEM']['text']
    assert 'Record both claims only for answer-relevant mutually exclusive observations' in system
    assert 'otherwise use an empty conflicts list' in system


@pytest.mark.parametrize('variant', ('R1_EVIDENCE', 'ALL5'))
def test_reflect_wire_anchor_is_scalar_without_dropping_coverage(variant):
    manifest = rules.prompt_manifest(config(variant))
    check = manifest['schemas']['reflect']['checks'][0]
    assert check['region_id'] == 'string'
    assert 'region_ids' not in check
    assert check['coverage_sufficient'] == 'boolean'
    assert 'all available Regions' in manifest['templates']['REFLECT_SYSTEM']['text']


def test_unknown_variants_and_flags_are_rejected():
    with pytest.raises(ValueError, match='Unknown rules-v2 variant'):
        rules.templates_for_config(config('BASE_COMPAT'))
    with pytest.raises(ValueError, match='Unknown rules-v2 flags'):
        rules.reflect_keys({'R6'})


class FakeSource:
    def __init__(self, pixels, width=64, height=64):
        self.pixels = pixels
        self.width, self.height = width, height
        self.reads = []

    def read(self, region):
        self.reads.append(region.region_id)
        return Image.fromarray(self.pixels, 'RGB')


def region(x=0, y=0, width=64, height=64):
    return Region('r', '/unused', x, y, width, height, 64, 64, 'physical', 5)


@pytest.mark.parametrize('x,y', [(64, 0), (0, 64), (-64, 0), (0, -64), (100, 100)])
def test_fully_outside_is_rejected_without_reading_image(x, y):
    source = FakeSource(np.full((64, 64, 3), 255, np.uint8))
    result = quality_for_region(source, region(x=x, y=y))
    assert result['hard_reject'] is True
    assert result['reason'] == 'fully_outside'
    assert result['valid_pixel_fraction'] == 0
    assert result['padding_fraction'] == 1
    assert result['colored_pixel_fraction'] is None
    assert source.reads == []


@pytest.mark.parametrize('rgb', [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_uniform_neutral_blank_is_rejected(rgb):
    image = np.full((64, 64, 3), rgb, np.uint8)
    result = quality_for_region(FakeSource(image), region())
    assert result['hard_reject'] is True
    assert result['reason'] == 'definite_blank'
    assert result['colored_pixel_fraction'] == 0
    assert result['flat_fraction'] == 1


def test_extremely_sparse_colored_tissue_is_preserved():
    image = np.full((64, 64, 3), 255, np.uint8)
    image[32, 32] = (130, 55, 150)
    result = quality_for_region(FakeSource(image), region())
    assert result['colored_pixel_fraction'] == 1 / 4096
    assert result['hard_reject'] is False


def test_partial_padding_is_recorded_but_colored_tissue_is_kept():
    image = np.full((64, 64, 3), 255, np.uint8)
    image[16:32, 16:32] = (180, 120, 170)
    result = quality_for_region(FakeSource(image), region(x=-32, y=-32))
    assert result['valid_pixel_fraction'] == .25
    assert result['padding_fraction'] == .75
    assert result['hard_reject'] is False


def test_uncolored_uncertain_textured_region_is_kept():
    image = np.indices((64, 64)).sum(axis=0) % 2 * 100 + 80
    pixels = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    result = quality_for_region(FakeSource(pixels), region())
    assert result['colored_pixel_fraction'] == 0
    assert result['neutral_fraction'] == 0
    assert result['flat_fraction'] == 0
    assert result['hard_reject'] is False


def test_partial_edge_tiles_use_pixel_weighted_flat_fraction():
    image = np.full((17, 17, 3), 120, np.uint8)
    result = quality_for_region(FakeSource(image), region())
    assert result['flat_fraction'] == 1


def test_partial_edge_tile_weights_are_not_equal_to_full_tile_weights():
    image = np.full((17, 17, 3), 120, np.uint8)
    checker = np.indices((16, 16)).sum(axis=0) % 2 * 100 + 80
    image[:16, :16] = checker[:, :, None]
    result = quality_for_region(FakeSource(image), region())
    assert result['flat_fraction'] == 33 / 289
    assert result['hard_reject'] is False


def test_channel_difference_does_not_wrap_unsigned_pixels():
    image = np.full((64, 64, 3), (250, 100, 245), np.uint8)
    result = quality_for_region(FakeSource(image), region())
    assert result['colored_pixel_fraction'] == 1
    assert result['hard_reject'] is False
