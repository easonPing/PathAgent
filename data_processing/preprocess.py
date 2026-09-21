"""Content-addressed image caches shared by all questions of one slide."""
import subprocess
import time
import numpy as np
from data_processing.common import (ROOT, atomic_json, digest, read_json, resolve,
                                    sample_seed, seed_everything, sha256_file)
from data_processing.regions import ImageSource, Region, whole_roi, wsi_region


def prepare_regions(sample, config, source_hash):
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
    key = digest({"models": config["models"], "generation": config["generation"], "seed": config["seed"]})
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
