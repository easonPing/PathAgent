"""Small, dependency-light primitives shared by preparation and inference."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_config(path="configs/reproduce.yaml"):
    import yaml
    with open(resolve(path), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction config schema")
    required = {"max_iterations": 5, "initial_ratio": .10, "replenish_ratio": .05,
                "question_description_topk": 5, "initial_magnification": 5,
                "zoom_query": "missing_info", "zoom_selected_children": 1}
    if any(config["algorithm"].get(k) != v for k, v in required.items()):
        raise ValueError("Algorithm differs from the frozen paper reproduction protocol")
    if any(config["generation"][k] is not False for k in ["enable_thinking", "final_do_sample", "summary_do_sample"]):
        raise ValueError("Generation flags differ from the frozen reproduction protocol")
    if any(config["generation"][k] != "checkpoint" for k in ["executor_sampling", "perceptor_sampling"]):
        raise ValueError("Predict/reflect/describe sampling must use the pinned checkpoint configuration")
    if config["slurm"]["shards"] != 16 or config["slurm"]["max_concurrent_gpus"] != 32:
        raise ValueError("Approved execution layout requires 16 shards per benchmark and at most 32 GPUs")
    return config


def code_hash():
    """Hash executed sources, including uncommitted changes, not just git HEAD."""
    files = [ROOT / "pathagent.py"]
    for folder in ["models", "data_processing", "data_preparation_script", "scripts", "eval"]:
        files.extend((ROOT / folder).rglob("*.py"))
    return digest({str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)})


def sample_seed(seed, sample_id, stage="agent"):
    return int(digest([seed, sample_id, stage])[:8], 16)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def read_manifest(path):
    data = read_json(resolve(path))
    samples = data["samples"]
    ids = [r["sample_id"] for r in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError("Manifest must be nonempty with unique sample IDs")
    if data.get("samples_hash") != digest(samples):
        raise ValueError("Manifest contents do not match samples_hash")
    return data


def write_manifest(path, samples, **metadata):
    if not samples:
        raise ValueError("Cannot write an empty manifest")
    ids = [r["sample_id"] for r in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    atomic_json(path, {"schema_version": 1, **metadata, "samples_hash": digest(samples),
                       "samples": samples})
