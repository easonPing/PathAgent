"""Read-only baseline geometry/PLIP and independently identified Generic caches."""
import fcntl
import os
import time
from functools import lru_cache
from pathlib import Path

from data_processing.common import atomic_json, digest, read_json, resolve, sha256_file
from data_processing.regions import Region
from models.rules_v2_prompts import templates_for_config


@lru_cache(maxsize=8)
def _manifest(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('rules-v2 input manifest changed')
    value = read_json(path)
    if value.get('schema_version') != 1 or value.get('kind') != 'frozen_initial_inputs':
        raise ValueError('Unsupported rules-v2 frozen inputs')
    return value


@lru_cache(maxsize=4096)
def _file(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen stage input changed: ' + path)
    return read_json(path)


@lru_cache(maxsize=4096)
def _image(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Original image changed: ' + path)


def frozen_inputs_v2(config):
    runtime = config['runtime']
    value = _manifest(str(resolve(runtime['rules_v2_initial_inputs_manifest'])),
                      runtime['rules_v2_initial_inputs_hash'])
    for key in ('seed', 'models', 'generation', 'preprocessing'):
        if config[key] != value['contract'][key]:
            raise ValueError('rules-v2 baseline contract mismatch: ' + key)
    if config['algorithm']['initial_magnification'] != value['contract']['initial_magnification']:
        raise ValueError('rules-v2 baseline scale differs')
    return value


def prepare_regions_v2(sample, config):
    value = frozen_inputs_v2(config)
    record = value['images'].get(str(sample['image_id']))
    if record is None or str(resolve(sample['image_path'])) != record['image_path']:
        raise ValueError('Image not in frozen rules-v2 inputs')
    for key in ('dataset', 'input_kind', 'assumed_mpp'):
        if sample.get(key) != record[key]:
            raise ValueError('Frozen image geometry differs: ' + key)
    _image(record['image_path'], record['image_sha256'])
    directory = Path(record['directory'])
    payload = _file(str(directory / 'regions.json'), record['regions_sha256'])
    if payload['image_sha256'] != record['image_sha256']:
        raise ValueError('Geometry provenance differs')
    regions = [Region(**item) for item in payload['regions']]
    if not regions or any(r.image_path != record['image_path'] for r in regions):
        raise ValueError('Invalid frozen Region coverage')
    return directory, regions


def prepare_observations_v2(directory, regions, backend, config):
    manifest = frozen_inputs_v2(config)
    matches = [(key, rec) for key, rec in manifest['images'].items()
               if rec['directory'] == str(directory)]
    if len(matches) != 1:
        raise ValueError('Unknown rules-v2 geometry directory')
    image_id, record = matches[0]
    base = _file(str(Path(directory) / 'observations.json'), record['observations_sha256'])
    ids = {r.region_id for r in regions}
    if ids != set(base['descriptions']) or ids != set(base['features']):
        raise ValueError('Frozen observations do not cover geometry')
    templates = templates_for_config(config)
    from hashlib import sha256
    hashes = {name: sha256(templates[name].encode()).hexdigest()
              for name in ('PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'PERCEPTOR_USER')}
    original_hashes = manifest['contract']['perceptor_prompt_hashes']
    common = {'image_id': image_id, 'image_sha256': record['image_sha256'],
              'baseline_observations': str(Path(directory) / 'observations.json'),
              'baseline_observations_sha256': record['observations_sha256'],
              'generic_prompt_hashes': hashes}
    actual_user = backend.perceptor_user('{metadata}', None, None)
    from models.context_prompts import PERCEPTOR_USER
    from models.paper_prompts import GENERIC_PROMPT
    baseline_user = PERCEPTOR_USER.format(metadata='{metadata}', instruction=GENERIC_PROMPT)
    if all(hashes[name] == original_hashes[name] for name in hashes) and actual_user == baseline_user:
        backend.generic_provenance = {**common, 'kind': 'baseline_cache',
                                      'directory': str(directory)}
        return base['descriptions'], base['features']

    # Original image-cache identity preserves the old per-region seed derivation.
    if not record.get('source_cache'):
        raise ValueError('New Generic generation requires the original seed provenance')
    generic_case = Path(record['source_cache']).name
    identity = {'image_sha256': record['image_sha256'], 'models': config['models']['perceptor'],
                'generation': config['generation'], 'seed': config['seed'],
                'generic_case_id': generic_case, 'prompt_hashes': hashes,
                'geometry_sha256': record['regions_sha256'],
                'effective_generic_user_template': actual_user,
                'resolved_perceptor_generation': getattr(backend, 'resolved_generation', {}).get('perceptor', {})}
    cache = resolve(config['runtime']['processed_root']) / 'generic' / digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    descriptions = {}
    generated_calls = []
    backend.set_case(generic_case)
    backend.records.clear()
    for region in regions:
        path = cache / (digest(region.to_dict()) + '.json')
        with open(str(path) + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                item = read_json(path)
                if item['identity_hash'] != digest(identity) or item['region'] != region.to_dict():
                    raise ValueError('Generic stage cache identity differs')
            else:
                description = backend.describe(region)
                item = {'identity_hash': digest(identity), 'region': region.to_dict(),
                        'description': description, 'model_call': backend.records[-1]}
                atomic_json(path, item)
                generated_calls.append(item['model_call'])
            descriptions[region.region_id] = item['description']
    backend.generic_provenance = {**common, 'kind': 'rules_v2_generic_cache',
                                  'directory': str(cache), 'identity': identity,
                                  'identity_hash': digest(identity),
                                  'generated_calls_this_process': len(generated_calls),
                                  'generated_seconds_this_process': sum(c.get('seconds', 0) for c in generated_calls)}
    # Cache records retain every Generic call; this file only describes work in this process.
    execution_id = [config['rules_v2']['variant'], image_id,
                    os.environ.get('SLURM_JOB_ID'), os.getpid()]
    atomic_json(cache / ('generated_calls_' + digest(execution_id) + '.json'),
                {'variant': config['rules_v2']['variant'], 'image_id': image_id,
                 'model_calls': generated_calls})
    atomic_json(cache / 'manifest.json', {**backend.generic_provenance,
                                        'region_count': len(regions),
                                        'seconds': time.monotonic() - started})
    return descriptions, base['features']
