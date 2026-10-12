"""Audit/copy archived experiments, then explicitly run ONE frozen replay shard.

No scheduler submission, source rewriting, cache rekeying, or automatic full run.
"""
import argparse
from collections import defaultdict
import copy
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

REPOSITORY = Path(__file__).resolve().parents[1]
BASELINE = REPOSITORY / "runs/submissions/20260929_v3_bcnb1000_cptac_run"
RULES_V0 = REPOSITORY / "runs/submissions/rules_v0_bcnb1000_20261006_234601"
DATASET = "bcnb_1000_sample"
V0_VARIANTS = ("O_ONLY", "D_ONLY", "R_ONLY", "M_ONLY", "F_ONLY", "ALL")
HASH_FOLDERS = ("models", "data_processing", "data_preparation_script", "scripts", "eval")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hash(source):
    source = Path(source)
    files = [source / "pathagent.py"]
    for folder in HASH_FOLDERS:
        files.extend((source / folder).rglob("*.py"))
    return digest({str(path.relative_to(source)): file_hash(path) for path in sorted(files)})


def checked_manifest(path):
    data = read(path)
    rows = data["samples"]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows) or data["samples_hash"] != digest(rows):
        raise ValueError("Invalid manifest: " + str(path))
    return data


def no_symlinks(path):
    path = Path(path)
    if path.is_symlink() or any(child.is_symlink() for child in path.rglob("*")):
        raise ValueError("Replay copies require ordinary files/directories, not symlinks: " + str(path))


def tree_hashes(path):
    path = Path(path)
    no_symlinks(path)
    return {str(child.relative_to(path)): file_hash(child) for child in sorted(path.rglob("*")) if child.is_file()}


def cache_key(sample, image_sha, config, code):
    return digest({"image": image_sha, "image_path": str(Path(sample["image_path"]).resolve()),
                   "kind": sample["input_kind"], "mpp": sample.get("assumed_mpp"),
                   "preprocessing": config["preprocessing"],
                   "initial_magnification": config["algorithm"]["initial_magnification"], "code": code})


def observations_key(config, prompts):
    return digest({"models": config["models"], "generation": config["generation"],
                   "seed": config["seed"], "protocol": prompts})


def environment_probe(executable):
    program = ("import json,sys,importlib.metadata as m; "
               "names=['torch','transformers','accelerate','bitsandbytes','qwen-vl-utils']; "
               "print(json.dumps({'executable':sys.executable,'packages':{n:m.version(n) for n in names}}))")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    value = subprocess.run([executable, "-c", program], env=env, check=True, capture_output=True, text=True)
    return json.loads(value.stdout)


def check_environment(expected, probe=None):
    actual = (probe or environment_probe)(expected["executable"])
    if Path(actual["executable"]).resolve() != Path(expected["executable"]).resolve():
        raise ValueError("Frozen Python executable differs")
    for name, version in expected.get("packages", {}).items():
        if actual["packages"].get(name) != version:
            raise ValueError("Frozen environment version differs: " + name)
    return actual


def check_assets(config, deep=False):
    records = {}
    models = dict(config["models"])
    segmenter = config.get("preprocessing", {}).get("segmentation_model")
    if segmenter:
        models["segmentation_model"] = segmenter
    for role, spec in models.items():
        directory = Path(spec["path"])
        lock_path = directory / "asset_lock.json"
        if not lock_path.exists():
            # Segmentation assets have not always used the LLM asset-lock schema.
            if role == "segmentation_model":
                filename = spec.get("filename")
                if not filename or not (directory / filename).is_file():
                    raise ValueError("Missing frozen segmentation asset")
                file = directory / filename
                records[role] = {"path": str(file), "bytes": file.stat().st_size,
                                 "sha256": file_hash(file) if deep else None}
                continue
            raise ValueError("Missing model asset lock: " + str(lock_path))
        lock = read(lock_path)
        if not lock.get("complete") or any(lock.get(key) != spec.get(key) for key in ("repo", "revision")):
            raise ValueError("Frozen asset identity differs: " + role)
        if role == "segmentation_model" and "files" not in lock:
            file = directory / lock["filename"]
            if not file.is_file() or (deep and file_hash(file) != lock["sha256"]):
                raise ValueError("Frozen segmentation asset differs")
            records[role] = {"asset_lock": str(lock_path), "asset_lock_sha256": file_hash(lock_path),
                             "path": str(file), "bytes": file.stat().st_size,
                             "content_hashes_verified": deep}
            continue
        for relative, info in lock["files"].items():
            file = directory / relative
            if not file.is_file() or file.stat().st_size != info["bytes"]:
                raise ValueError("Frozen asset missing/truncated: " + str(file))
            if deep and file_hash(file) != info["sha256"]:
                raise ValueError("Frozen asset content differs: " + str(file))
        records[role] = {"asset_lock": str(lock_path), "asset_lock_sha256": file_hash(lock_path),
                         "files": len(lock["files"]), "content_hashes_verified": deep}
    return records


def resolve_origins(version, variant=None, origin=None):
    if version == "baseline":
        if variant not in (None, "BASELINE"):
            raise ValueError("baseline only accepts variant BASELINE")
        return {"BASELINE": Path(origin or BASELINE).resolve()}
    if version != "rules-v0":
        raise ValueError("Unknown frozen version")
    if variant is not None and variant not in V0_VARIANTS:
        raise ValueError("Unknown rules-v0 variant")
    campaign = Path(origin or RULES_V0).resolve()
    plan = read(campaign / "plan.json")
    names = (variant,) if variant else V0_VARIANTS
    return {name: Path(plan["variants"][name]["root"]).resolve() for name in names}


def audit_origin(origin, version, deep_cache=False, deep_assets=False, probe=None):
    origin = Path(origin).resolve()
    source = origin / "source"
    full_path = origin / DATASET / "full.json"
    full = checked_manifest(full_path)
    samples = {row["sample_id"]: row for row in full["samples"]}
    image_samples = {str(row["image_id"]): row for row in full["samples"]}
    shard_dirs = sorted((origin / DATASET / "full").glob("shard_*"), key=lambda path: int(path.name[6:]))
    expected_count = 16 if version == "baseline" else 10
    if [path.name for path in shard_dirs] != ["shard_" + str(i) for i in range(expected_count)]:
        raise ValueError("Frozen shard layout differs: expected " + str(expected_count))
    reference, prompts, environment, gpu = None, None, None, None
    inventory, shard_manifests, seen = {}, {}, set()
    for directory in shard_dirs:
        lock = read(directory / "run_lock.json")
        if lock.get("run_hash") != digest({key: value for key, value in lock.items() if key != "run_hash"}):
            raise ValueError("Frozen run_lock invalid: " + str(directory))
        current = {key: lock[key] for key in ("config", "code_hash", "protocol", "prompt_manifest_hash")}
        if reference is not None and current != reference:
            raise ValueError("Mixed frozen shard identities")
        reference = current
        current_prompts = read(directory / "prompt_manifest.json")
        if digest(current_prompts) != lock["prompt_manifest_hash"]:
            raise ValueError("Frozen prompt manifest differs")
        prompts = current_prompts
        path = origin / DATASET / "manifests" / (directory.name + ".json")
        shard = checked_manifest(path)
        if shard["samples_hash"] != lock["samples_hash"]:
            raise ValueError("Frozen shard manifest differs from run lock")
        for sample in shard["samples"]:
            sid = sample["sample_id"]
            if sid in seen or samples.get(sid) != sample:
                raise ValueError("Duplicate/mismatched frozen shard sample: " + sid)
            seen.add(sid)
        shard_manifests[directory.name] = str(path)
        env = read(directory / "environment.json")
        env = {key: env.get(key) for key in ("executable", "packages", "trident_python")}
        if environment is not None and env != environment:
            raise ValueError("Mixed frozen environments")
        environment = env
        profile = read(directory / "profile.json")
        current_gpu = {key: profile[key] for key in ("gpu", "gpu_memory_bytes")}
        if gpu is not None and gpu != current_gpu:
            raise ValueError("Mixed frozen GPU models")
        gpu = current_gpu
        for image, info in read(directory / "image_inventory.json").items():
            if image in inventory and inventory[image] != info:
                raise ValueError("Conflicting frozen image inventory: " + image)
            inventory[image] = info
    if seen != set(samples) or set(inventory) != set(image_samples):
        raise ValueError("Incomplete frozen manifests/inventory")
    if source_hash(source) != reference["code_hash"]:
        raise ValueError("Frozen source hash mismatch")
    config = reference["config"]
    if config["slurm"]["shards"] != expected_count:
        raise ValueError("Frozen configuration shard count differs")
    observed_environment = check_environment(environment, probe)
    assets = check_assets(config, deep_assets)
    obs_key = observations_key(config, prompts)
    cache_records = {}
    for image, info in sorted(inventory.items()):
        directory = Path(info["observation_cache"]).resolve()
        sample = image_samples[image]
        key = cache_key(sample, info["sha256"], config, reference["code_hash"])
        regions = read(directory / "regions.json")
        if directory.name != key or regions.get("key") != key or regions.get("image_sha256") != info["sha256"]:
            raise ValueError("Frozen region cache key differs: " + image)
        active = directory / ("observations_" + obs_key + ".json")
        obs = read(active)
        region_ids = {region["region_id"] for region in regions["regions"]}
        if set(obs["descriptions"]) != region_ids or set(obs["features"]) != region_ids:
            raise ValueError("Frozen observations are incomplete: " + image)
        for rid in region_ids:
            if not (directory / ("description_" + digest([obs_key, rid]) + ".json")).is_file():
                raise ValueError("Missing generic description provenance: " + image)
        relative = directory.relative_to(Path(config["runtime"]["processed_root"]).resolve())
        cache_records[image] = {"origin": str(directory), "relative": str(relative), "key": key,
                                "observations_key": obs_key, "image_path": sample["image_path"],
                                "image_sha256": info["sha256"],
                                "file_hashes": tree_hashes(directory) if deep_cache else None}
    smoke_path = origin / DATASET / "smoke.json"
    smoke = checked_manifest(smoke_path)
    if smoke.get("purpose") != "smoke" or any(samples.get(row["sample_id"]) != row for row in smoke["samples"]):
        raise ValueError("Frozen smoke is not a valid full-manifest subset")
    return {"origin": str(origin), "source": str(source), **reference, "prompts": prompts,
            "environment": environment, "observed_environment": observed_environment, "assets": assets, "gpu": gpu,
            "full_manifest": str(full_path), "validation_manifest": str(smoke_path), "shard_manifests": shard_manifests,
            "samples_hash": full["samples_hash"], "sample_count": len(samples), "shards": expected_count,
            "cache_records": cache_records, "image_count": len(inventory), "deep_cache": deep_cache,
            "deep_assets": deep_assets, "note": "CPU identity audit; no model inference or job submission."}


def relocated_config(original, processed_root, manifest_root, account=None):
    config = copy.deepcopy(original)
    config["runtime"]["processed_root"] = str(Path(processed_root).resolve())
    config["runtime"]["manifest_root"] = str(Path(manifest_root).resolve())
    if account is not None:
        config["slurm"]["account"] = account
    check_config_relocation(original, config)
    return config


def check_config_relocation(original, relocated):
    before, after = copy.deepcopy(original), copy.deepcopy(relocated)
    for key in ("processed_root", "manifest_root"):
        before["runtime"].pop(key, None)
        after["runtime"].pop(key, None)
    # Account is a scheduler identity, not an inference algorithm parameter.
    before["slurm"].pop("account", None)
    after["slurm"].pop("account", None)
    if before != after:
        raise ValueError("Replay attempted to change scientific or unapproved resource configuration")


def copy_checked_tree(source, destination, exclusions=()):
    source, destination = Path(source), Path(destination)
    no_symlinks(source)
    if destination.exists():
        raise ValueError("Replay destination already exists: " + str(destination))
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*exclusions))
    hashes = tree_hashes(destination)
    expected = {str(path.relative_to(source)): file_hash(path) for path in source.rglob("*") if path.is_file()
                and not any(fnmatch.fnmatch(part, pattern) for part in path.relative_to(source).parts for pattern in exclusions)}
    if hashes != expected:
        raise ValueError("Copied files differ from frozen origin: " + str(source))
    return hashes


def prepare(root, version, variant=None, origin=None, account=None, deep_assets=False, probe=None):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a new independent replay root")
    origins = resolve_origins(version, variant, origin)
    audits = {name: audit_origin(path, version, False, deep_assets, probe) for name, path in origins.items()}
    for audit in audits.values():
        protected = [Path(audit["origin"]), *[Path(item["origin"]) for item in audit["cache_records"].values()]]
        if any(root == path or path in root.parents for path in protected):
            raise ValueError("Replay root must not be inside frozen data")
    root.mkdir(parents=True)
    plan = {"schema_version": 1, "root": str(root), "version": version,
            "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(), "arms": {},
            "mode": "frozen_source_and_byte_identical_cache_replay", "automatic_full_run": False,
            "scheduler_submission": False, "cache_keys_rewritten": False}
    for name, audit in audits.items():
        base = root / "arms" / name
        copied_source = base / "source"
        source_files = copy_checked_tree(audit["source"], copied_source, ("__pycache__", ".cache", "*.pyc"))
        if source_hash(copied_source) != audit["code_hash"]:
            raise ValueError("Copied frozen source code_hash differs")
        data = base / DATASET
        data.mkdir(parents=True)
        shutil.copy2(audit["full_manifest"], data / "full.json")
        shutil.copy2(audit["validation_manifest"], data / "smoke.json")
        for shard, path in audit["shard_manifests"].items():
            target = data / "manifests" / (shard + ".json")
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
        copied_caches = {}
        for image, item in audit["cache_records"].items():
            target = base / "processed" / item["relative"]
            # Entire directories, not just regions/active observations; preserve every key and byte.
            hashes = copy_checked_tree(item["origin"], target)
            copied_caches[image] = {**item, "copy": str(target), "file_hashes": hashes}
        config = relocated_config(audit["config"], base / "processed", base / "manifests", account)
        config_path = base / "config.json"
        write(config_path, config)  # JSON is valid YAML for the unchanged frozen loader.
        record = {"root": str(base), "source": str(copied_source), "source_hash": audit["code_hash"],
                  "source_files": source_files, "config": str(config_path), "config_hash": digest(config),
                  "original_config": audit["config"], "prompt_manifest_hash": audit["prompt_manifest_hash"],
                  "protocol": audit["protocol"], "origin": audit["origin"], "environment": audit["environment"],
                  "assets": audit["assets"], "gpu": audit["gpu"], "shards": audit["shards"],
                  "full_manifest": str(data / "full.json"), "validation_manifest": str(data / "smoke.json"),
                  "shard_manifests": {name: str(data / "manifests" / (name + ".json")) for name in audit["shard_manifests"]},
                  "samples_hash": audit["samples_hash"], "caches": copied_caches}
        record["manifest_hashes"] = {path: file_hash(path) for path in [record["full_manifest"], record["validation_manifest"], *record["shard_manifests"].values()]}
        plan["arms"][name] = record
    write(root / "replay_plan.json", plan)
    (root / "replay_plan.sha256").write_text(digest(plan) + "\n", encoding="utf-8")
    return plan


def validate_plan(root, arm=None, probe=None):
    root = Path(root).resolve()
    plan = read(root / "replay_plan.json")
    if (root / "replay_plan.sha256").read_text().strip() != digest(plan) or Path(plan["root"]).resolve() != root:
        raise ValueError("Replay plan identity differs")
    if arm is None:
        if len(plan["arms"]) != 1:
            raise ValueError("Specify --variant for a multi-arm replay")
        arm = next(iter(plan["arms"]))
    record = plan["arms"][arm]
    config = read(record["config"])
    if digest(config) != record["config_hash"]:
        raise ValueError("Replay config changed")
    check_config_relocation(record["original_config"], config)
    if tree_hashes(record["source"]) != record["source_files"] or source_hash(record["source"]) != record["source_hash"]:
        raise ValueError("Replay frozen source changed")
    for path, sha in record["manifest_hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Replay manifest changed")
    for item in record["caches"].values():
        if tree_hashes(item["copy"]) != item["file_hashes"]:
            raise ValueError("Replay copied cache changed: " + item["copy"])
    check_environment(record["environment"], probe)
    current_assets = check_assets(config, False)
    for role, old in record["assets"].items():
        for key in ("asset_lock_sha256", "bytes"):
            if key in old and old[key] != current_assets[role].get(key):
                raise ValueError("Replay asset metadata changed: " + role)
    return plan, arm, record


def run_one(root, stage, shard=None, variant=None, dry_run=False, probe=None, runner=None):
    if stage not in ("validation", "full"):
        raise ValueError("Explicit --stage validation|full is required")
    plan, name, record = validate_plan(root, variant, probe)
    if stage == "full":
        if type(shard) is not int or not 0 <= shard < record["shards"]:
            raise ValueError("Full replay requires one explicit valid --shard")
        manifest = record["shard_manifests"]["shard_" + str(shard)]
        output = Path(record["root"]) / DATASET / "full" / ("shard_" + str(shard))
    else:
        if shard is not None:
            raise ValueError("Validation does not accept --shard")
        manifest = record["validation_manifest"]
        output = Path(record["root"]) / DATASET / "validation"
    command = [record["environment"]["executable"], "pathagent.py", "--config", record["config"],
               "--manifest", manifest, "--output", str(output)]
    if stage == "validation":
        command.append("--smoke")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PATHAGENT_EXPECT_CODE_HASH=record["source_hash"],
                       PATHAGENT_EXPECT_GPU_NAME=record["gpu"]["gpu"],
                       PATHAGENT_EXPECT_GPU_MEMORY_BYTES=str(record["gpu"]["gpu_memory_bytes"]),
                       PATHAGENT_GPU_FLEET_LOCK=str(Path(root).resolve() / "gpu_fleet.json"))
    receipt = {"variant": name, "stage": stage, "shard": shard, "command": command,
               "cwd": record["source"], "output": str(output), "dry_run": dry_run,
               "note": "One direct frozen process; no scheduler submission and no implicit next shard."}
    if not dry_run:
        result = (runner or subprocess.run)(command, cwd=record["source"], env=environment, check=False)
        receipt["returncode"] = result.returncode
        write(Path(root) / "receipts" / f"{name}-{stage}-{shard}.json", receipt)
        if result.returncode:
            raise RuntimeError("Frozen replay exited with code " + str(result.returncode))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare"):
        sub = subs.add_parser(command)
        sub.add_argument("--version", required=True, choices=("baseline", "rules-v0"))
        sub.add_argument("--variant")
        sub.add_argument("--origin", help="Explicit frozen archive root; normally use the built-in versions")
        sub.add_argument("--deep-assets", action="store_true")
        if command == "audit":
            sub.add_argument("--deep-cache", action="store_true")
        else:
            sub.add_argument("--root", required=True)
            sub.add_argument("--account")
    run = subs.add_parser("run")
    run.add_argument("--root", required=True)
    run.add_argument("--variant")
    run.add_argument("--stage", required=True, choices=("validation", "full"))
    run.add_argument("--shard", type=int)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        results = {}
        for name, origin in resolve_origins(args.version, args.variant, args.origin).items():
            value = audit_origin(origin, args.version, args.deep_cache, args.deep_assets)
            results[name] = {key: value[key] for key in ("origin", "source", "code_hash", "prompt_manifest_hash", "sample_count", "image_count", "shards", "deep_cache", "deep_assets", "note")}
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        value = prepare(args.root, args.version, args.variant, args.origin, args.account, args.deep_assets)
        print(json.dumps({"replay_plan": str(Path(args.root).resolve() / "replay_plan.json"), "arms": list(value["arms"])}))
    else:
        print(json.dumps(run_one(args.root, args.stage, args.shard, args.variant, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
