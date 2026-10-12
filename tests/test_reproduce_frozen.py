"""Frozen replay may relocate storage, but never rewrite scientific identities."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from slurm import reproduce_frozen as replay


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_probe(executable):
    return {"executable": executable, "packages": {}}


def make_archive(tmp_path, version="baseline"):
    origin = tmp_path / "origin"
    source = origin / "source"
    source.mkdir(parents=True)
    (source / "pathagent.py").write_text("# frozen entrypoint\n")
    (source / "models").mkdir()
    (source / "models/prompts.py").write_text("PROMPT = 'unchanged'\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/irrelevant.pyc").write_bytes(b"compiled")
    asset = tmp_path / "weights"
    asset.mkdir()
    (asset / "model.bin").write_bytes(b"frozen weights")
    put(asset / "asset_lock.json", {"repo": "test/model", "revision": "abc", "complete": True,
        "files": {"model.bin": {"bytes": 14, "sha256": replay.file_hash(asset / "model.bin")}}})
    count = 16 if version == "baseline" else 10
    config = {"protocol": "frozen", "seed": 128, "models": {"executor": {"path": str(asset), "repo": "test/model", "revision": "abc"}},
              "generation": {"tokens": 5}, "preprocessing": {"level0_patch_size": 4096},
              "algorithm": {"initial_magnification": 5, "zoom_magnifications": [10, 20, 40]},
              "runtime": {"processed_root": str(tmp_path / "cache"), "manifest_root": "/old/manifests"},
              "slurm": {"shards": count, "account": "old-account", "cpus_per_task": 8}}
    prompts = {"protocol": "frozen", "templates": {"test": "unchanged"}}
    code = replay.source_hash(source)
    rows = []
    for index in range(count):
        image = tmp_path / f"image{index}.jpg"
        image.write_bytes(b"source image " + str(index).encode())
        sample = {"sample_id": f"bcnb:{index}", "image_id": str(index), "image_path": str(image),
                  "input_kind": "wsi", "assumed_mpp": .5, "dataset": "bcnb", "question": "test", "choices": {"A": "yes", "B": "no"}, "ground_truth_label": "A"}
        rows.append(sample)
        key = replay.cache_key(sample, replay.file_hash(image), config, code)
        directory = tmp_path / "cache/bcnb" / key
        obs_key = replay.observations_key(config, prompts)
        region = {"region_id": "region", "image_path": str(image)}
        put(directory / "regions.json", {"key": key, "image_sha256": replay.file_hash(image), "regions": [region]})
        put(directory / ("observations_" + obs_key + ".json"), {"descriptions": {"region": "unchanged"}, "features": {"region": [1, 2]}})
        put(directory / ("description_" + replay.digest([obs_key, "region"]) + ".json"), {"region": region, "description": "unchanged"})
        put(directory / "trident/nested.json", {"path": "/old/must/not/be/rewritten"})
        # Full-cache preservation also includes files irrelevant to active observations.
        (directory / "extra.pyc").write_bytes(b"cache byte payload")
        shard = origin / replay.DATASET / "full" / f"shard_{index}"
        manifest = {"samples": [sample], "samples_hash": replay.digest([sample]), "purpose": "shard", "dataset": "bcnb"}
        put(origin / replay.DATASET / "manifests" / f"shard_{index}.json", manifest)
        identity = {"config": config, "code_hash": code, "protocol": "frozen", "prompt_manifest_hash": replay.digest(prompts), "samples_hash": manifest["samples_hash"]}
        put(shard / "run_lock.json", {**identity, "run_hash": replay.digest(identity)})
        put(shard / "prompt_manifest.json", prompts)
        put(shard / "environment.json", {"executable": sys.executable, "packages": {}, "trident_python": "/unused"})
        put(shard / "profile.json", {"gpu": "test GPU", "gpu_memory_bytes": 96 * 1024**3})
        put(shard / "image_inventory.json", {str(index): {"observation_cache": str(directory), "path": str(image), "sha256": replay.file_hash(image)}})
    put(origin / replay.DATASET / "full.json", {"samples": rows, "samples_hash": replay.digest(rows), "purpose": "full", "dataset": "bcnb"})
    put(origin / replay.DATASET / "smoke.json", {"samples": rows[:2], "samples_hash": replay.digest(rows[:2]), "purpose": "smoke", "dataset": "bcnb"})
    return origin, config


def test_prepare_preserves_source_shards_and_every_cache_byte(tmp_path):
    origin, config = make_archive(tmp_path)
    old_hashes = replay.tree_hashes(origin)
    target = tmp_path / "replay"
    plan = replay.prepare(target, "baseline", origin=origin, account="new-account", probe=fake_probe, deep_assets=True)
    arm = plan["arms"]["BASELINE"]
    assert arm["shards"] == 16 and len(arm["shard_manifests"]) == 16
    moved = replay.read(arm["config"])
    assert moved["algorithm"] == config["algorithm"] and moved["seed"] == 128
    assert moved["slurm"]["account"] == "new-account"
    assert arm["source_hash"] == replay.source_hash(origin / "source")
    for item in arm["caches"].values():
        assert replay.tree_hashes(item["origin"]) == replay.tree_hashes(item["copy"])
        assert "extra.pyc" in item["file_hashes"]
        assert Path(item["origin"]).name == Path(item["copy"]).name
        assert (Path(item["copy"]) / "trident/nested.json").read_text() == '{"path": "/old/must/not/be/rewritten"}'
    assert replay.tree_hashes(origin) == old_hashes
    assert not list(target.rglob("results/*.json"))
    replay.validate_plan(target, probe=fake_probe)


def test_run_only_one_explicit_shard_from_frozen_source(tmp_path):
    origin, _ = make_archive(tmp_path)
    target = tmp_path / "replay"
    plan = replay.prepare(target, "baseline", origin=origin, probe=fake_probe)
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    receipt = replay.run_one(target, "full", shard=7, probe=fake_probe, runner=runner)
    assert len(calls) == 1 and receipt["shard"] == 7
    assert calls[0][1]["cwd"] == plan["arms"]["BASELINE"]["source"]
    assert calls[0][0][1] == "pathagent.py" and "--smoke" not in calls[0][0]
    assert calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert calls[0][1]["env"]["PATHAGENT_EXPECT_CODE_HASH"] == plan["arms"]["BASELINE"]["source_hash"]
    smoke = replay.run_one(target, "validation", dry_run=True, probe=fake_probe)
    assert "--smoke" in smoke["command"]
    with pytest.raises(ValueError, match="explicit"):
        replay.run_one(target, "full", probe=fake_probe, runner=runner)
    assert len(calls) == 1


@pytest.mark.parametrize("what", ["source", "cache", "manifest", "config"])
def test_mutation_rejected_before_any_run(tmp_path, what):
    origin, _ = make_archive(tmp_path)
    target = tmp_path / "replay"
    plan = replay.prepare(target, "baseline", origin=origin, probe=fake_probe)
    arm = plan["arms"]["BASELINE"]
    paths = {"source": Path(arm["source"]) / "pathagent.py", "cache": Path(next(iter(arm["caches"].values()))["copy"]) / "regions.json",
             "manifest": Path(arm["full_manifest"]), "config": Path(arm["config"])}
    paths[what].write_text("{}")
    with pytest.raises(ValueError):
        replay.run_one(target, "validation", dry_run=True, probe=fake_probe)


def test_scientific_relocation_rejected_and_environment_checked(tmp_path):
    _, config = make_archive(tmp_path)
    bad = copy.deepcopy(config)
    bad["algorithm"]["zoom_magnifications"] = [40]
    with pytest.raises(ValueError, match="scientific"):
        replay.check_config_relocation(config, bad)
    with pytest.raises(ValueError, match="version"):
        replay.check_environment({"executable": sys.executable, "packages": {"torch": "old"}}, fake_probe)


def test_rules_v0_optional_variant_and_original_10_shards(tmp_path):
    origin, _ = make_archive(tmp_path, "rules-v0")
    campaign = tmp_path / "v0"
    put(campaign / "plan.json", {"variants": {"M_ONLY": {"root": str(origin)}}})
    plan = replay.prepare(tmp_path / "replay", "rules-v0", "M_ONLY", campaign, probe=fake_probe)
    assert plan["arms"]["M_ONLY"]["shards"] == 10
    with pytest.raises(ValueError, match="Unknown"):
        replay.resolve_origins("rules-v0", "new-rule", campaign)
    with pytest.raises(ValueError, match="new independent"):
        replay.prepare(tmp_path / "replay", "rules-v0", "M_ONLY", campaign, probe=fake_probe)


def test_audit_is_read_only_and_rejects_changed_source(tmp_path):
    origin, _ = make_archive(tmp_path)
    before = replay.tree_hashes(origin)
    result = replay.audit_origin(origin, "baseline", deep_cache=True, probe=fake_probe)
    assert result["sample_count"] == 16 and result["deep_cache"]
    assert replay.tree_hashes(origin) == before
    (origin / "source/models/prompts.py").write_text("PROMPT='changed'\n")
    with pytest.raises(ValueError, match="source hash"):
        replay.audit_origin(origin, "baseline", probe=fake_probe)


def test_copy_does_not_follow_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "linked").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        replay.copy_checked_tree(source, tmp_path / "dest")
