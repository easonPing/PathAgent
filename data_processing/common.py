"""Small, dependency-light primitives shared by preparation and inference."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from models.protocols import PAPER_PROTOCOL, CONTEXT_PROTOCOL, RULES_V2_PROTOCOL, protocol_name, protocol_manifest
from models.region_selection import selection_policy

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


def load_config(path="configs/reproduce_v1.yaml"):
    import yaml
    with open(resolve(path), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction config schema")
    protocol = protocol_name(config)
    policy = selection_policy(config['algorithm'])
    required = {"max_iterations": 5, "initial_ratio": .10, "replenish_ratio": .10 if policy == 'max5' else .05,
                "question_description_topk": 5, "initial_magnification": 5,
                "zoom_query": "missing_info", "zoom_selected_children": 5 if policy == 'max5' or protocol == RULES_V2_PROTOCOL else 1}
    if any(config["algorithm"].get(k) != v for k, v in required.items()):
        raise ValueError("Algorithm differs from the frozen paper reproduction protocol")
    if type(config['algorithm'].get('zoom_return_to_loop', False)) is not bool:
        raise ValueError('zoom_return_to_loop must be a boolean')
    if protocol == RULES_V2_PROTOCOL:
        from models.rules_v2_prompts import flags_for_variant
        flags_for_variant(config.get('rules_v2', {}).get('variant'))
        if not config['algorithm'].get('zoom_return_to_loop'):
            raise ValueError('rules-v2 requires post-Zoom review')
        if config['runtime'].get('initial_inputs_manifest'):
            raise ValueError('rules-v2 uses a separate staged-input contract')
        v2_path = config['runtime'].get('rules_v2_initial_inputs_manifest')
        v2_hash = config['runtime'].get('rules_v2_initial_inputs_hash')
        if (not isinstance(v2_path, str) or not v2_path or not isinstance(v2_hash, str)
                or len(v2_hash) != 64 or any(c not in '0123456789abcdef' for c in v2_hash)):
            raise ValueError('rules-v2 requires frozen geometry and SHA256')
    snapshot = config['runtime'].get('initial_inputs_manifest')
    snapshot_hash = config['runtime'].get('initial_inputs_hash')
    if (snapshot is None) != (snapshot_hash is None):
        raise ValueError('Frozen initial input manifest and hash must be supplied together')
    if snapshot is not None:
        if (not isinstance(snapshot, str) or not snapshot
                or not isinstance(snapshot_hash, str)
                or len(snapshot_hash) != 64
                or any(c not in '0123456789abcdef' for c in snapshot_hash)):
            raise ValueError('Frozen initial inputs require a manifest and SHA256')
    if any(config["generation"][k] is not False for k in ["enable_thinking", "final_do_sample", "summary_do_sample"]):
        raise ValueError("Generation flags differ from the frozen reproduction protocol")
    if any(config["generation"][k] != "checkpoint" for k in ["executor_sampling", "perceptor_sampling"]):
        raise ValueError("Predict/reflect/describe sampling must use the pinned checkpoint configuration")
    shards = config["slurm"]["shards"]
    concurrency = config["slurm"].get("max_concurrent_gpus")
    if (type(shards) is not int or shards not in {6, 10, 16, 128}
            or (concurrency is not None and (type(concurrency) is not int or concurrency != 32))):
        raise ValueError("Approved layouts use 6, 10, 16 or 128 shards; GPU concurrency is 32 or null (scheduler controlled)")
    if protocol in (PAPER_PROTOCOL, CONTEXT_PROTOCOL, RULES_V2_PROTOCOL):
        caps = {'step_a_tokens': 1024, 'step_b_tokens': 512, 'step_c_tokens': 512,
                'generic_description_tokens': 1024, 'question_description_tokens': 2048,
                'final_tokens': 2048, 'summary_tokens': 1024}
        if any(config['generation'].get(k) != v for k, v in caps.items()):
            raise ValueError('Output budgets differ from the named 2x protocol')
        if config['generation'].get('retries') != 1:
            raise ValueError('The paper protocol allows one format retry')
    expected_dtypes = {'executor': 'bfloat16', 'perceptor': 'bfloat16', 'navigator': 'float32'}
    for role, dtype in expected_dtypes.items():
        if config['models'][role]['dtype'] != dtype or config['models'][role].get('quantization'):
            raise ValueError('Local comparison requires BF16 Executor/Perceptor and FP32 PLIP')
    if any(device != 'cuda:0' for device in config['runtime'].get('devices', {}).values()):
        raise ValueError('Local comparison supports one GPU per process (cuda:0)')
    return config


def run_identity(config, source_hash, samples_hash):
    return {'config': config, 'code_hash': source_hash, 'samples_hash': samples_hash,
            'protocol': protocol_name(config), 'prompt_manifest_hash': digest(protocol_manifest(config))}


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
