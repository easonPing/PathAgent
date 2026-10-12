"""Content-addressed image caches shared by all questions of one slide."""
import subprocess
import time
import numpy as np
from functools import lru_cache
from pathlib import Path
from data_processing.common import (ROOT, atomic_json, digest, read_json, resolve,
                                    sample_seed, seed_everything, sha256_file)
from data_processing.regions import ImageSource, Region, whole_roi, wsi_region
from models.protocols import protocol_manifest


@lru_cache(maxsize=8)
def _initial_manifest(path, expected_hash):
    if sha256_file(path) != expected_hash:
        raise ValueError('Frozen initial input manifest changed')
    value = read_json(path)
    if value.get('schema_version') != 1 or value.get('kind') != 'frozen_initial_inputs':
        raise ValueError('Unsupported frozen initial input manifest')
    return value


@lru_cache(maxsize=4096)
def _initial_file(path, expected_hash):
    if sha256_file(path) != expected_hash:
        raise ValueError('Frozen initial input file changed: ' + path)
    return read_json(path)


@lru_cache(maxsize=4096)
def _initial_image(path, expected_hash):
    if sha256_file(path) != expected_hash:
        raise ValueError('Frozen original image changed: ' + path)


def frozen_initial_inputs(config):
    """Optional read-only input adapter; legacy cache identity remains untouched."""
    path = config['runtime'].get('initial_inputs_manifest')
    if path is None:
        return None
    value = _initial_manifest(str(resolve(path)), config['runtime']['initial_inputs_hash'])
    contract = value['contract']
    for key in ('seed', 'models', 'generation', 'preprocessing'):
        if config[key] != contract[key]:
            raise ValueError('Frozen initial input contract mismatch: ' + key)
    if config['algorithm']['initial_magnification'] != contract['initial_magnification']:
        raise ValueError('Frozen initial input scale mismatch')
    prompts = protocol_manifest(config)['templates']
    required = {'PERCEPTOR_SYSTEM', 'GENERIC_PROMPT', 'QUESTION_PROMPT', 'PERCEPTOR_USER', 'ZOOM_MISSING_USER'}
    if set(contract['perceptor_prompt_hashes']) != required:
        raise ValueError('Incomplete frozen Perceptor prompt contract')
    for name, expected in contract['perceptor_prompt_hashes'].items():
        if prompts[name]['sha256'] != expected:
            raise ValueError('Frozen Perceptor prompt mismatch: ' + name)
    return value


def prepare_regions(sample, config, source_hash):
    frozen = frozen_initial_inputs(config)
    if frozen is not None:
        record = frozen['images'].get(str(sample['image_id']))
        if record is None or str(Path(resolve(sample['image_path'])).resolve()) != record['image_path']:
            raise ValueError('Image not in frozen initial inputs')
        if any(key not in record or sample.get(key) != record[key]
               for key in ('dataset', 'input_kind', 'assumed_mpp')):
            raise ValueError('Frozen image geometry contract mismatch')
        _initial_image(record['image_path'], record['image_sha256'])
        directory = Path(record['directory'])
        value = _initial_file(str(directory / 'regions.json'), record['regions_sha256'])
        if value['image_sha256'] != record['image_sha256']:
            raise ValueError('Frozen image provenance mismatch')
        regions = [Region(**r) for r in value['regions']]
        if not regions or any(r.image_path != record['image_path'] for r in regions):
            raise ValueError('Frozen region image identity mismatch')
        return directory, regions
    image_hash = sha256_file(sample["image_path"])
    key = digest({"image": image_hash, "image_path": str(resolve(sample["image_path"])), "kind": sample["input_kind"],
                  "mpp": sample["assumed_mpp"], "preprocessing": config["preprocessing"],
                  "initial_magnification": config["algorithm"]["initial_magnification"],
                  "code": source_hash})
    directory = resolve(config["runtime"]["processed_root"]) / sample["dataset"] / key
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "regions.json"
    if cache.exists():
        record = read_json(cache)
        return directory, [Region(**r) for r in record["regions"]]
    started = time.monotonic()
    with ImageSource(sample["image_path"], sample["assumed_mpp"]) as source:
        if sample["input_kind"] == "roi":
            regions = [whole_roi(source)]
        else:
            if source.mpp is None:
                raise ValueError("WSI has no physical scale and no declared MPP assumption")
            output = directory / "coords.json"
            request = {"image_path": source.path, "reader_type": "openslide" if source.slide else "image",
                       "mpp": source.mpp, "job_dir": str(directory / "trident"), "output": str(output),
                       "preprocessing": config["preprocessing"],
                       "workers": config["runtime"]["workers"],
                       "seed": sample_seed(config["seed"], image_hash, "segmentation")}
            request_path = directory / "request.json"
            atomic_json(request_path, request)
            subprocess.run([str(resolve(config["preprocessing"]["trident_python"])),
                            str(ROOT / "scripts/trident_coords.py"), "--request", str(request_path)], check=True)
            regions = [wsi_region(source, x, y, config["preprocessing"]["level0_patch_size"],
                                  config["algorithm"]["initial_magnification"])
                       for x, y in read_json(output)["coords"]]
        record = {"key": key, "image_sha256": image_hash, "width": source.width,
                  "height": source.height, "mpp": source.mpp, "scale_source": source.mpp_source,
                  "seconds": time.monotonic() - started, "regions": [r.to_dict() for r in regions]}
    atomic_json(cache, record)
    return directory, regions


def prepare_observations(directory, regions, backend, config):
    frozen = frozen_initial_inputs(config)
    if frozen is not None:
        records = [r for r in frozen['images'].values() if r['directory'] == str(directory)]
        if len(records) != 1:
            raise ValueError('Unknown frozen observation directory')
        record = records[0]
        value = _initial_file(str(directory / 'observations.json'), record['observations_sha256'])
        ids = {r.region_id for r in regions}
        if ids != set(value['descriptions']) or ids != set(value['features']):
            raise ValueError('Frozen observation coverage mismatch')
        return value['descriptions'], value['features']
    key = digest({"models": config["models"], "generation": config["generation"], "seed": config["seed"],
                  "protocol": protocol_manifest(config)})
    cache = directory / f"observations_{key}.json"
    if cache.exists():
        value = read_json(cache)
        return value["descriptions"], value["features"]
    started = time.monotonic()
    descriptions = {}
    backend.set_case(directory.name)
    backend.records.clear()
    for region in regions:
        partial = directory / ("description_" + digest([key, region.region_id]) + ".json")
        if partial.exists():
            descriptions[region.region_id] = read_json(partial)["description"]
        else:
            descriptions[region.region_id] = backend.describe(region)
            atomic_json(partial, {"description": descriptions[region.region_id],
                                  "model_call": backend.records[-1], "region": region.to_dict()})
    features = backend.encode_regions(regions)
    value = {"descriptions": descriptions,
             "features": {r.region_id: np.asarray(f).tolist() for r, f in zip(regions, features)},
             "seconds": time.monotonic() - started}
    atomic_json(cache, value)
    return value["descriptions"], value["features"]
