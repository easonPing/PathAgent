"""CPU-only integrity contracts for opt-in frozen Generic inputs/features."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import yaml

from data_processing import preprocess
from data_processing.common import atomic_json, digest, load_config, read_json, sha256_file
from data_processing.regions import Region


ROOT = Path(__file__).resolve().parents[1]
_CACHES = (preprocess._initial_manifest, preprocess._initial_file, preprocess._initial_image)
PERCEPTOR_TEMPLATES = ('GENERIC_PROMPT', 'QUESTION_PROMPT', 'PERCEPTOR_SYSTEM',
                       'PERCEPTOR_USER', 'ZOOM_MISSING_USER')


def clear_frozen_caches():
    for cached in _CACHES:
        cached.cache_clear()


@pytest.fixture(autouse=True)
def isolated_frozen_caches():
    # Real workers assume frozen files are immutable. Tests that alter them
    # start with fresh readers instead of reusing an lru_cache result.
    clear_frozen_caches()
    yield
    clear_frozen_caches()


def refresh_manifest(state):
    atomic_json(state.manifest_path, state.manifest)
    state.config['runtime']['initial_inputs_hash'] = sha256_file(state.manifest_path)
    clear_frozen_caches()


def replace_frozen_file(state, name, payload):
    path = state.directory / name
    atomic_json(path, payload)
    key = {'regions.json': 'regions_sha256',
           'observations.json': 'observations_sha256'}[name]
    state.manifest['images'][state.sample['image_id']][key] = sha256_file(path)
    refresh_manifest(state)


@pytest.fixture
def frozen(tmp_path):
    config = load_config(ROOT / 'configs/reproduce_v3.yaml')
    image_path = tmp_path / 'original.png'
    Image.new('RGB', (32, 16), (130, 80, 160)).save(image_path)
    image_path = image_path.resolve()
    sample = {'image_id': 'image-one', 'image_path': str(image_path),
              'dataset': 'bcnb', 'input_kind': 'wsi', 'assumed_mpp': .5}
    regions = [Region(f'r{i}', str(image_path), i * 16, 0, 16, 16, 8, 8,
                      'physical', 5, .5, 'assumed') for i in range(2)]
    directory = tmp_path / 'frozen-image'
    directory.mkdir()
    region_payload = {'image_sha256': sha256_file(image_path), 'width': 32, 'height': 16,
                      'regions': [region.to_dict() for region in regions]}
    observations = {'descriptions': {'r0': 'generic observation zero',
                                      'r1': 'generic observation one'},
                    'features': {'r0': [1.0, 0.0], 'r1': [0.0, 1.0]}}
    atomic_json(directory / 'regions.json', region_payload)
    atomic_json(directory / 'observations.json', observations)
    prompts = preprocess.protocol_manifest(config)['templates']
    contract = {key: deepcopy(config[key])
                for key in ('seed', 'models', 'generation', 'preprocessing')}
    contract.update(initial_magnification=config['algorithm']['initial_magnification'],
                    perceptor_prompt_hashes={name: prompts[name]['sha256']
                                            for name in PERCEPTOR_TEMPLATES})
    manifest = {'schema_version': 1, 'kind': 'frozen_initial_inputs',
                'contract': contract,
                'images': {sample['image_id']: {
                    'image_path': str(image_path), 'image_sha256': region_payload['image_sha256'],
                    'dataset': sample['dataset'], 'input_kind': sample['input_kind'],
                    'assumed_mpp': sample['assumed_mpp'],
                    'directory': str(directory),
                    'regions_sha256': sha256_file(directory / 'regions.json'),
                    'observations_sha256': sha256_file(directory / 'observations.json'),
                }}}
    manifest_path = tmp_path / 'frozen-inputs.json'
    config['runtime']['initial_inputs_manifest'] = str(manifest_path)
    state = SimpleNamespace(config=config, sample=sample, image_path=image_path,
                            directory=directory, regions=regions,
                            region_payload=region_payload, observations=observations,
                            manifest=manifest, manifest_path=manifest_path)
    refresh_manifest(state)
    return state


class NoBackendCalls:
    def __getattr__(self, name):
        raise AssertionError('Frozen inputs must not invoke backend.' + name)


@pytest.mark.parametrize('config_name', ['reproduce_v1.yaml', 'reproduce_v2.yaml', 'reproduce_v3.yaml'])
def test_loading_legacy_config_does_not_insert_opt_in_flags(config_name):
    path = ROOT / 'configs' / config_name
    original = yaml.safe_load(path.read_text())
    loaded = load_config(path)
    assert loaded == original
    assert 'initial_inputs_manifest' not in loaded['runtime']
    assert 'initial_inputs_hash' not in loaded['runtime']
    assert 'zoom_return_to_loop' not in loaded['algorithm']
    assert preprocess.frozen_initial_inputs(loaded) is None


def test_no_flag_uses_legacy_region_and_observation_cache(tmp_path):
    config = load_config(ROOT / 'configs/reproduce_v3.yaml')
    config['runtime']['processed_root'] = str(tmp_path / 'legacy-processed')
    path = tmp_path / 'legacy.png'
    Image.new('RGB', (24, 12), 'purple').save(path)
    sample = {'image_id': 'legacy', 'image_path': str(path), 'dataset': 'test',
              'input_kind': 'roi', 'assumed_mpp': None}
    directory, regions = preprocess.prepare_regions(sample, config, 'legacy-source')
    assert len(regions) == 1 and regions[0].region_id == 'roi'
    assert regions[0].scale_kind == 'relative'
    record = read_json(directory / 'regions.json')
    assert record['image_sha256'] == sha256_file(path)
    assert directory.name == record['key']

    class RecordingBackend:
        def __init__(self):
            self.records, self.calls = [], []

        def set_case(self, case):
            self.calls.append(('set_case', case))

        def describe(self, region):
            self.calls.append(('describe', region.region_id))
            self.records.append({'stage': 'describe', 'region': region.to_dict()})
            return 'legacy generic description'

        def encode_regions(self, current):
            self.calls.append(('encode_regions', len(current)))
            return np.array([[1.0, 0.0] for _ in current])

    backend = RecordingBackend()
    descriptions, features = preprocess.prepare_observations(directory, regions, backend, config)
    assert descriptions == {'roi': 'legacy generic description'}
    assert features == {'roi': [1.0, 0.0]}
    assert backend.calls == [('set_case', directory.name), ('describe', 'roi'), ('encode_regions', 1)]
    key = digest({'models': config['models'], 'generation': config['generation'],
                  'seed': config['seed'], 'protocol': preprocess.protocol_manifest(config)})
    assert (directory / ('observations_' + key + '.json')).is_file()
    assert preprocess.prepare_regions(sample, config, 'legacy-source') == (directory, regions)
    assert preprocess.prepare_observations(directory, regions, NoBackendCalls(), config) == (descriptions, features)


def test_frozen_path_reuses_exact_inputs_without_image_preprocessing_or_backend(frozen, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Frozen inputs must not re-extract regions or run segmentation')
    monkeypatch.setattr(preprocess, 'ImageSource', unexpected)
    monkeypatch.setattr(preprocess.subprocess, 'run', unexpected)
    before = {p.name: p.read_bytes() for p in frozen.directory.iterdir()}
    directory, regions = preprocess.prepare_regions(frozen.sample, frozen.config, 'new-source-hash')
    assert directory == frozen.directory and regions == frozen.regions
    assert preprocess.prepare_regions(frozen.sample, frozen.config, 'different-source-hash') == (directory, regions)
    descriptions, features = preprocess.prepare_observations(directory, regions, NoBackendCalls(), frozen.config)
    assert descriptions == frozen.observations['descriptions']
    assert features == frozen.observations['features']
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_image_symlink_and_relative_alias_match_canonical_frozen_identity(frozen, tmp_path, monkeypatch):
    alias = tmp_path / 'alias.png'
    alias.symlink_to(frozen.image_path)
    sample = dict(frozen.sample, image_path=str(alias))
    assert preprocess.prepare_regions(sample, frozen.config, 'source')[1] == frozen.regions
    # resolve(relative) is repository-relative, not cwd-relative; substitute
    # its configured root so this alias tests the same public path contract.
    monkeypatch.setattr('data_processing.common.ROOT', tmp_path)
    sample['image_path'] = 'alias.png'
    assert preprocess.prepare_regions(sample, frozen.config, 'source')[1] == frozen.regions


@pytest.mark.parametrize('change', ['unknown_id', 'different_image_path'])
def test_frozen_image_identity_rejects_a_different_sample(frozen, tmp_path, change):
    sample = dict(frozen.sample)
    if change == 'unknown_id':
        sample['image_id'] = 'unlisted'
    else:
        different = tmp_path / 'different.png'
        different.write_bytes(frozen.image_path.read_bytes())
        sample['image_path'] = str(different)
    with pytest.raises(ValueError, match='Image not in frozen initial inputs'):
        preprocess.prepare_regions(sample, frozen.config, 'source')


@pytest.mark.parametrize('key', ['seed', 'models', 'generation', 'preprocessing'])
def test_frozen_contract_mismatch_rejects_reuse(frozen, key):
    config = deepcopy(frozen.config)
    if key == 'seed':
        config[key] += 1
    elif key == 'models':
        config[key]['executor']['revision'] = 'another-checkpoint-revision'
    elif key == 'generation':
        config[key]['generic_description_tokens'] += 1
    else:
        config[key]['overlap'] += 1
    with pytest.raises(ValueError, match='Frozen initial input contract mismatch: ' + key):
        preprocess.frozen_initial_inputs(config)


def test_frozen_initial_magnification_mismatch(frozen):
    config = deepcopy(frozen.config)
    config['algorithm']['initial_magnification'] = 10
    with pytest.raises(ValueError, match='Frozen initial input scale mismatch'):
        preprocess.frozen_initial_inputs(config)


@pytest.mark.parametrize('name', PERCEPTOR_TEMPLATES)
def test_changed_initial_perceptor_prompt_hash_is_rejected(frozen, monkeypatch, name):
    prompts = deepcopy(preprocess.protocol_manifest(frozen.config))
    prompts['templates'][name]['sha256'] = '0' * 64
    monkeypatch.setattr(preprocess, 'protocol_manifest', lambda config: prompts)
    with pytest.raises(ValueError, match='Frozen Perceptor prompt mismatch: ' + name):
        preprocess.frozen_initial_inputs(frozen.config)


def test_executor_prompt_and_zoom_control_changes_do_not_invalidate_generic_contract(frozen, monkeypatch):
    prompts = deepcopy(preprocess.protocol_manifest(frozen.config))
    prompts['templates']['PREDICT_SYSTEM']['sha256'] = '1' * 64
    prompts['variant'] = 'ALL4'
    prompts['post_zoom_reflect'] = True
    monkeypatch.setattr(preprocess, 'protocol_manifest', lambda config: prompts)
    config = deepcopy(frozen.config)
    config['algorithm']['zoom_return_to_loop'] = True
    assert preprocess.frozen_initial_inputs(config) == frozen.manifest
    directory, regions = preprocess.prepare_regions(frozen.sample, config, 'new-executor-source')
    assert preprocess.prepare_observations(directory, regions, NoBackendCalls(), config) == (
        frozen.observations['descriptions'], frozen.observations['features'])


def test_changed_manifest_bytes_are_rejected(frozen):
    frozen.manifest_path.write_bytes(frozen.manifest_path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Frozen initial input manifest changed'):
        preprocess.frozen_initial_inputs(frozen.config)


@pytest.mark.parametrize('key,value', [('schema_version', 2), ('kind', 'ordinary-cache')])
def test_unsupported_manifest_is_rejected_even_with_matching_hash(frozen, key, value):
    frozen.manifest[key] = value
    refresh_manifest(frozen)
    with pytest.raises(ValueError, match='Unsupported frozen initial input manifest'):
        preprocess.frozen_initial_inputs(frozen.config)


def test_changed_original_image_bytes_are_rejected_before_cache_reuse(frozen):
    preprocess.prepare_regions(frozen.sample, frozen.config, 'source')
    frozen.image_path.write_bytes(frozen.image_path.read_bytes() + b'changed-original')
    clear_frozen_caches()  # A fresh worker must verify pixels, not just the old region JSON.
    with pytest.raises(ValueError, match='Frozen original image changed'):
        preprocess.prepare_regions(frozen.sample, frozen.config, 'new-source')


@pytest.mark.parametrize('filename', ['regions.json', 'observations.json'])
def test_changed_frozen_file_bytes_are_rejected(frozen, filename):
    path = frozen.directory / filename
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Frozen initial input file changed'):
        if filename == 'regions.json':
            preprocess.prepare_regions(frozen.sample, frozen.config, 'source')
        else:
            preprocess.prepare_observations(frozen.directory, frozen.regions, NoBackendCalls(), frozen.config)


def test_region_image_provenance_must_match_the_manifest(frozen):
    payload = deepcopy(frozen.region_payload)
    payload['image_sha256'] = '0' * 64
    replace_frozen_file(frozen, 'regions.json', payload)
    with pytest.raises(ValueError, match='Frozen image provenance mismatch'):
        preprocess.prepare_regions(frozen.sample, frozen.config, 'source')


@pytest.mark.parametrize('change', ['empty', 'wrong_image'])
def test_frozen_region_list_has_valid_image_identity(frozen, change):
    payload = deepcopy(frozen.region_payload)
    if change == 'empty':
        payload['regions'] = []
    else:
        payload['regions'][0]['image_path'] = '/wrong/original.png'
    replace_frozen_file(frozen, 'regions.json', payload)
    with pytest.raises(ValueError, match='Frozen region image identity mismatch'):
        preprocess.prepare_regions(frozen.sample, frozen.config, 'source')


@pytest.mark.parametrize('field', ['descriptions', 'features'])
@pytest.mark.parametrize('change', ['missing', 'extra', 'renamed'])
def test_observation_coverage_must_exactly_match_region_ids(frozen, field, change):
    payload = deepcopy(frozen.observations)
    if change == 'missing':
        payload[field].pop('r0')
    elif change == 'extra':
        payload[field]['unrequested'] = payload[field]['r0']
    else:
        payload[field]['renamed'] = payload[field].pop('r0')
    replace_frozen_file(frozen, 'observations.json', payload)
    with pytest.raises(ValueError, match='Frozen observation coverage mismatch'):
        preprocess.prepare_observations(frozen.directory, frozen.regions, NoBackendCalls(), frozen.config)


def test_observations_reject_a_different_region_subset(frozen):
    with pytest.raises(ValueError, match='Frozen observation coverage mismatch'):
        preprocess.prepare_observations(frozen.directory, frozen.regions[:1], NoBackendCalls(), frozen.config)


@pytest.mark.parametrize('change', ['unknown', 'duplicate'])
def test_observation_directory_must_identify_exactly_one_frozen_image(frozen, tmp_path, change):
    directory = frozen.directory
    if change == 'unknown':
        directory = tmp_path / 'not-in-manifest'
    else:
        frozen.manifest['images']['duplicate-image'] = deepcopy(frozen.manifest['images'][frozen.sample['image_id']])
        refresh_manifest(frozen)
    with pytest.raises(ValueError, match='Unknown frozen observation directory'):
        preprocess.prepare_observations(directory, frozen.regions, NoBackendCalls(), frozen.config)


@pytest.mark.parametrize('runtime_patch,message', [
    ({'initial_inputs_manifest': '/unused/manifest.json'}, 'must be supplied together'),
    ({'initial_inputs_hash': 'a' * 64}, 'must be supplied together'),
    ({'initial_inputs_manifest': '', 'initial_inputs_hash': 'a' * 64}, 'require a manifest and SHA256'),
    ({'initial_inputs_manifest': '/unused/manifest.json', 'initial_inputs_hash': 'short'}, 'require a manifest and SHA256'),
    ({'initial_inputs_manifest': '/unused/manifest.json', 'initial_inputs_hash': 'g' * 64}, 'require a manifest and SHA256'),
    ({'initial_inputs_manifest': '/unused/manifest.json', 'initial_inputs_hash': 123}, 'require a manifest and SHA256'),
    ({'initial_inputs_manifest': 123, 'initial_inputs_hash': 'a' * 64}, 'require a manifest and SHA256'),
])
def test_load_config_requires_frozen_manifest_and_valid_hash_together(tmp_path, runtime_patch, message):
    config = load_config(ROOT / 'configs/reproduce_v3.yaml')
    config['runtime'].update(runtime_patch)
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_load_config_preserves_explicit_frozen_flags(frozen, tmp_path):
    config = deepcopy(frozen.config)
    config['algorithm']['zoom_return_to_loop'] = True
    path = tmp_path / 'explicit.yaml'
    path.write_text(yaml.safe_dump(config))
    assert load_config(path) == config


@pytest.mark.parametrize('change', ['missing', 'extra', 'empty'])
def test_perceptor_contract_requires_exactly_all_five_template_hashes(frozen, change):
    hashes = frozen.manifest['contract']['perceptor_prompt_hashes']
    if change == 'missing':
        hashes.pop('ZOOM_MISSING_USER')
    elif change == 'extra':
        hashes['PREDICT_SYSTEM'] = 'a' * 64
    else:
        hashes.clear()
    refresh_manifest(frozen)
    with pytest.raises(ValueError, match='Incomplete frozen Perceptor prompt contract'):
        preprocess.frozen_initial_inputs(frozen.config)


@pytest.mark.parametrize('field,value', [('dataset', 'other'), ('input_kind', 'roi'), ('assumed_mpp', .25)])
def test_changed_sample_geometry_contract_is_rejected(frozen, field, value):
    sample = dict(frozen.sample)
    sample[field] = value
    with pytest.raises(ValueError, match='Frozen image geometry contract mismatch'):
        preprocess.prepare_regions(sample, frozen.config, 'source')


@pytest.mark.parametrize('field', ['dataset', 'input_kind', 'assumed_mpp'])
def test_missing_frozen_image_geometry_field_is_rejected(frozen, field):
    frozen.manifest['images'][frozen.sample['image_id']].pop(field)
    refresh_manifest(frozen)
    with pytest.raises(ValueError, match='Frozen image geometry contract mismatch'):
        preprocess.prepare_regions(frozen.sample, frozen.config, 'source')
