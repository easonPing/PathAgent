from copy import deepcopy
from pathlib import Path

import pytest

from test_frozen_initial_inputs_v1 import frozen
from data_processing.common import atomic_json, digest, sample_seed, sha256_file
from data_processing.rules_v2_inputs import prepare_observations_v2, prepare_regions_v2
from models.rules_v2_prompts import flags_for_variant, templates_for_config


@pytest.fixture
def state(frozen, tmp_path):
    config = deepcopy(frozen.config)
    config['protocol'] = 'pathology-rules-v2'
    config['rules_v2'] = {'variant': 'R1_EVIDENCE'}
    config['algorithm'].update(zoom_selected_children=5, zoom_return_to_loop=True)
    config['runtime'].pop('initial_inputs_manifest')
    config['runtime'].pop('initial_inputs_hash')
    config['runtime'].update(rules_v2_initial_inputs_manifest=str(frozen.manifest_path),
                             processed_root=str(tmp_path / 'v2-cache'))
    frozen.manifest['images']['image-one']['source_cache'] = str(tmp_path / 'original-seed-case')
    atomic_json(frozen.manifest_path, frozen.manifest)
    config['runtime']['rules_v2_initial_inputs_hash'] = sha256_file(frozen.manifest_path)
    frozen.config = config
    return frozen


class GenericBackend:
    def __init__(self, config):
        self.config = config
        self.records, self.calls = [], []
        self.resolved_generation = {'perceptor': {'do_sample': True}}

    def perceptor_user(self, metadata, question=None, missing_info=None):
        templates = templates_for_config(self.config)
        user = templates['PERCEPTOR_USER'].format(metadata=metadata,
                                                 instruction=templates['GENERIC_PROMPT'])
        if 'R2' in flags_for_variant(self.config['rules_v2']['variant']):
            user += '\nInput modality: H&E'
        return user

    def set_case(self, case):
        self.case_id = case

    def describe(self, region):
        self.calls.append(region.region_id)
        seed = sample_seed(self.config['seed'], self.case_id,
                           'describe:' + region.metadata() + 'NoneNone:0')
        self.records.append({'stage': 'describe', 'seed': seed, 'seconds': 2,
                             'raw_response': 'new generic ' + region.region_id})
        return self.records[-1]['raw_response']


def test_baseline_generic_reused_while_question_prompt_changes(state):
    state.config['rules_v2']['variant'] = 'R3_ROI_VALIDITY'
    backend = GenericBackend(state.config)
    before = {p.name: sha256_file(p) for p in state.directory.iterdir() if p.is_file()}
    directory, regions = prepare_regions_v2(state.sample, state.config)
    descriptions, features = prepare_observations_v2(directory, regions, backend, state.config)
    assert descriptions == state.observations['descriptions']
    assert features == state.observations['features']
    assert not backend.calls
    assert backend.generic_provenance['kind'] == 'baseline_cache'
    assert before == {p.name: sha256_file(p) for p in state.directory.iterdir() if p.is_file()}


def test_r2_and_all_share_new_generic_with_original_seed_identity(state):
    state.config['rules_v2']['variant'] = 'R2_DIFFERENTIAL'
    backend = GenericBackend(state.config)
    first, features = prepare_observations_v2(state.directory, state.regions, backend, state.config)
    assert backend.case_id == 'original-seed-case'
    assert backend.calls == ['r0', 'r1']
    assert backend.generic_provenance['generated_calls_this_process'] == 2
    config_all = deepcopy(state.config)
    config_all['rules_v2']['variant'] = 'ALL5'
    other = GenericBackend(config_all)
    second, other_features = prepare_observations_v2(state.directory, state.regions, other, config_all)
    assert second == first
    assert other_features == features == state.observations['features']
    assert not other.calls
    assert backend.generic_provenance['identity_hash'] == other.generic_provenance['identity_hash']


def test_effective_message_change_cannot_reuse_new_generic(state):
    state.config['rules_v2']['variant'] = 'R2_DIFFERENTIAL'
    first = GenericBackend(state.config)
    prepare_observations_v2(state.directory, state.regions, first, state.config)
    second = GenericBackend(state.config)
    old_user = second.perceptor_user
    second.perceptor_user = lambda *args: old_user(*args) + '\nNew wrapper'
    prepare_observations_v2(state.directory, state.regions, second, state.config)
    assert second.calls == ['r0', 'r1']
    assert first.generic_provenance['identity_hash'] != second.generic_provenance['identity_hash']


def test_missing_original_seed_provenance_rejects_new_generic(state):
    state.config['rules_v2']['variant'] = 'R2_DIFFERENTIAL'
    del state.manifest['images']['image-one']['source_cache']
    atomic_json(state.manifest_path, state.manifest)
    state.config['runtime']['rules_v2_initial_inputs_hash'] = sha256_file(state.manifest_path)
    with pytest.raises(ValueError, match='seed provenance'):
        prepare_observations_v2(state.directory, state.regions, GenericBackend(state.config), state.config)


def test_changed_scientific_contract_does_not_reuse_frozen_geometry(state):
    state.config['seed'] += 1
    with pytest.raises(ValueError, match='contract mismatch'):
        prepare_regions_v2(state.sample, state.config)


def test_loaded_new_generic_identity_is_verified(state):
    from data_processing.common import read_json
    state.config['rules_v2']['variant'] = 'R2_DIFFERENTIAL'
    backend = GenericBackend(state.config)
    prepare_observations_v2(state.directory, state.regions, backend, state.config)
    cache = Path(backend.generic_provenance['directory'])
    path = cache / (digest(state.regions[0].to_dict()) + '.json')
    value = read_json(path)
    value['identity_hash'] = 'bad'
    atomic_json(path, value)
    with pytest.raises(ValueError, match='identity differs'):
        prepare_observations_v2(state.directory, state.regions, GenericBackend(state.config), state.config)
