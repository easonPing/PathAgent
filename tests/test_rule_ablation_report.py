"""Reporting must preserve experimental identities and never count missing as regression."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from slurm import report_rule_ablations as report


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def manifest(path, samples):
    put(path, {"dataset": "bcnb", "samples_hash": report.digest(samples), "samples": samples})


def sample_rows():
    rows = []
    for index, (image, task, gold, choices) in enumerate([
        ("w1", "Tumor", "A", {"A": "IDC", "B": "ILC"}),
        ("w1", "Tumor", "B", {"A": "IDC", "B": "ILC"}),
        ("w2", "Histological grading", "B", {"A": "Grade 1", "B": "Grade 3"}),
        ("w3", "ER", "A", {"A": "Positive", "B": "Negative"}),
    ]):
        rows.append({"dataset": "bcnb", "sample_id": f"bcnb:{index}", "image_id": image,
                     "image_path": "/unused/" + image, "question": "Classify", "source": "",
                     "task": task, "choices": choices, "ground_truth_label": gold, "ground_truth": gold})
    return rows


def make_arm(root, samples, answers, name="original", split=False):
    base = root / "bcnb_1000_sample"
    groups = [samples[:2], samples[2:]] if split else [samples]
    config = {"models": {"executor": {"revision": "frozen"}}, "generation": {"do_sample": False},
              "seed": 128, "variant": name}
    prompt = {"protocol": "test-protocol", "templates": {"prompt": name}}
    for index, group in enumerate(groups):
        shard = base / "full" / f"shard_{index}"
        manifest(base / "manifests" / f"shard_{index}.json", group)
        identity = {"config": config, "code_hash": "frozen-source", "samples_hash": report.digest(group),
                    "protocol": "test-protocol", "prompt_manifest_hash": report.digest(prompt)}
        lock = {**identity, "run_hash": report.digest(identity)}
        put(shard / "run_lock.json", lock)
        put(shard / "prompt_manifest.json", prompt)
        put(shard / "profile.json", {"run_hash": lock["run_hash"], "seconds": 7})
        for sample in group:
            sid = sample["sample_id"]
            if sid not in answers:
                continue
            answer = answers[sid]
            result = {"sample_id": sid, "image_id": sample["image_id"], "status": "ok",
                      "pred_answer": answer, "explanation": "Observed architecture supports the answer.",
                      "run_hash": lock["run_hash"], "protocol": lock["protocol"], "agent_seconds": 2,
                      "process": [{"iteration": 1, "action": "conclude", "findings": [],
                                   "decision": {"answer": sample["ground_truth_label"], "sufficient": "Yes"}}],
                      "model_calls": [{"stage": "predict", "input_tokens": 10, "output_tokens": 2, "seconds": .1,
                                       "raw_response": json.dumps({"answer": sample["ground_truth_label"], "thinking_steps": "test"})},
                                      {"stage": "reflect", "input_tokens": 11, "output_tokens": 3, "seconds": .2,
                                       "raw_response": '{"sufficient": "Yes"}'},
                                      {"stage": "final", "input_tokens": 12, "output_tokens": 4, "seconds": .3,
                                       "raw_response": json.dumps({"answer": answer, "explanation": "test"})}]}
            put(shard / "results" / (sid.replace(":", "_") + ".json"), result)
    return base


@pytest.fixture
def campaign(tmp_path):
    samples = sample_rows()
    baseline = tmp_path / "old"
    variant = tmp_path / "variant"
    make_arm(baseline, samples, {"bcnb:0": "A", "bcnb:1": "A", "bcnb:2": "B", "bcnb:3": "A"}, split=True)
    make_arm(variant, samples, {"bcnb:0": "B", "bcnb:1": "B", "bcnb:2": "None of the above"}, name="O_ONLY", split=True)
    manifest(tmp_path / "full.json", samples)
    put(tmp_path / "plan.json", {"baseline_root": str(baseline),
        "datasets": {"bcnb_1000_sample": {"full_manifest": str(tmp_path / "full.json"), "baseline_expected_correct": 3}},
        "variants": {"O_ONLY": {"root": str(variant)}}})
    return tmp_path, samples, baseline, variant


def test_incomplete_denominator_pairing_and_candidate_examples(campaign):
    root, _, _, _ = campaign
    value, summary, transitions, bad, trajectories = report.build_report(root, bootstrap=100)
    arm = value["datasets"]["bcnb_1000_sample"]["arms"]["O_ONLY"]
    metrics = arm["metrics"]["overall"]
    assert not value["is_final"] and value["status"] == "incomplete_not_final"
    assert (metrics["expected"], metrics["received"], metrics["correct"], metrics["invalid"], metrics["missing"]) == (4, 3, 1, 1, 1)
    assert metrics["accuracy_percent"] == 25
    paired = arm["paired_with_baseline"]["overall"]
    assert paired["transitions"] == {"correct_correct": 0, "correct_wrong": 2, "wrong_correct": 1, "wrong_wrong": 0}
    assert paired["paired_received"] == 3 and paired["bootstrap"]["status"] == "withheld_incomplete"
    missing = next(row for row in transitions if row["variant"] == "O_ONLY" and row["sample_id"] == "bcnb:3")
    assert missing["correct"] is None and missing["transition"] is None
    assert not any(row["variant"] == "O_ONLY" and row["sample_id"] == "bcnb:3" for row in bad)
    assert all(not row["candidate_is_pixel_truth"] for row in bad)
    assert trajectories["bcnb_1000_sample"]["O_ONLY"]["qa_call_usage"]["raw_calls"] == 9
    assert trajectories["bcnb_1000_sample"]["BASELINE"]["generic_cache_generation"]["status"] == "skipped"
    assert value["datasets"]["bcnb_1000_sample"]["baseline_verification"]["status"] == "verified"


def test_complete_outputs_and_frozen_baseline_not_merged(campaign):
    root, samples, baseline, variant = campaign
    make_arm(variant, samples, {"bcnb:0": "B", "bcnb:1": "B", "bcnb:2": "None of the above", "bcnb:3": "A"}, name="O_ONLY", split=True)
    values = report.build_report(root, bootstrap=100)
    report.write_report(root, *values)
    assert values[0]["complete"]
    assert {path.name for path in root.glob("report*")} == {"report.json", "report.zh.md"}
    assert all((root / name).exists() for name in ("summary.csv", "transitions.csv", "bad_subset.jsonl", "trajectory_metrics.json"))
    arm = values[0]["datasets"]["bcnb_1000_sample"]["arms"]["O_ONLY"]
    assert arm["paired_with_baseline"]["overall"]["bootstrap"]["delta_accuracy_percent"] == -25
    assert values[0]["datasets"]["bcnb_1000_sample"]["arms"]["BASELINE"]["metrics"]["overall"]["correct"] == 3
    plan = report.read_json(root / "plan.json")
    plan["variants"]["O_ONLY"]["root"] = str(baseline)
    put(root / "plan.json", plan)
    with pytest.raises(ValueError, match="distinct"):
        report.build_report(root, bootstrap=100)


@pytest.mark.parametrize("corruption", ["duplicate", "run_hash", "mixed_config", "mixed_prompt", "outside", "manifest", "baseline"])
def test_identity_corruption_is_rejected(campaign, corruption):
    root, _, _, variant = campaign
    shard = variant / "bcnb_1000_sample/full/shard_0"
    path = shard / "results/bcnb_0.json"
    result = report.read_json(path)
    if corruption == "duplicate":
        put(shard / "results/duplicate.json", result)
    elif corruption == "run_hash":
        result["run_hash"] = "wrong"
        put(path, result)
    elif corruption in ("mixed_config", "mixed_prompt"):
        lock = report.read_json(shard / "run_lock.json")
        if corruption == "mixed_config":
            lock["config"]["variant"] = "other"
        else:
            prompt = {"protocol": "different"}
            put(shard / "prompt_manifest.json", prompt)
            lock["prompt_manifest_hash"] = report.digest(prompt)
        lock["run_hash"] = report.digest({key: value for key, value in lock.items() if key != "run_hash"})
        put(shard / "run_lock.json", lock)
        profile = report.read_json(shard / "profile.json")
        profile["run_hash"] = lock["run_hash"]
        put(shard / "profile.json", profile)
        for item in (shard / "results").glob("*.json"):
            row = report.read_json(item)
            row["run_hash"] = lock["run_hash"]
            put(item, row)
    elif corruption == "outside":
        result["sample_id"] = "bcnb:outside"
        put(path, result)
    elif corruption == "manifest":
        value = report.read_json(root / "full.json")
        value["samples"][0]["ground_truth_label"] = "B"
        put(root / "full.json", value)
    elif corruption == "baseline":
        plan = report.read_json(root / "plan.json")
        plan["datasets"]["bcnb_1000_sample"]["baseline_expected_correct"] = 4
        put(root / "plan.json", plan)
    with pytest.raises(ValueError):
        report.build_report(root, bootstrap=20)


def test_semantic_classes_do_not_depend_on_option_letter():
    samples = sample_rows()[:2]
    samples[1]["choices"] = {"A": "ILC", "B": "IDC"}
    samples[1]["ground_truth_label"] = "A"
    results = {row["sample_id"]: {"status": "ok", "pred_answer": "A"} for row in samples}
    metrics = report.group_metrics(samples, results)
    assert metrics["class_support"] == {"Tumor::IDC": 1, "Tumor::ILC": 1}
    assert metrics["balanced_accuracy_percent"] == 100
    # Strict scorer rejects a label paired with the wrong option body.
    assert report.answer_label("A. ILC", samples[0]["choices"]) is None


def test_bootstrap_repeats_whole_wsi_with_unequal_question_counts():
    import numpy as np
    samples = sample_rows()
    for index, row in enumerate(samples):
        row["image_id"] = "large" if index < 3 else "small"
    baseline = {row["sample_id"]: {"status": "ok", "pred_answer": "invalid"} for row in samples}
    variant = {row["sample_id"]: {"status": "ok", "pred_answer": row["ground_truth_label"] if index < 3 else "invalid"}
               for index, row in enumerate(samples)}
    value = report.paired_bootstrap(samples, baseline, variant, iterations=10000, seed=128)
    assert value["clusters"] == 2 and value["delta_accuracy_percent"] == 75
    rng = np.random.default_rng(128)
    # Each draw includes both QAs/WSI multiplicity; the large WSI carries 3 QAs.
    draws = rng.integers(0, 2, size=(10000, 2))
    n_large = (draws == 0).sum(axis=1)
    exact = 100 * 3 * n_large / (3 * n_large + (2 - n_large))
    assert value["delta_ci95_percent"] == np.quantile(exact, [.025, .975]).tolist()
    assert report.paired_bootstrap(samples, baseline, variant, 10000, 128) == value


def test_generic_cache_key_and_deduplication(campaign):
    root, samples, _, variant = campaign
    directory = root / "shared-cache"
    regions = [{"region_id": "r1"}, {"region_id": "r2"}]
    put(directory / "regions.json", {"regions": regions})
    shard = variant / "bcnb_1000_sample/full/shard_0"
    lock = report.read_json(shard / "run_lock.json")
    prompt = report.read_json(shard / "prompt_manifest.json")
    config = lock["config"]
    key = report.digest({"models": config["models"], "generation": config["generation"], "seed": config["seed"], "protocol": prompt})
    for region in regions:
        path = directory / ("description_" + report.digest([key, region["region_id"]]) + ".json")
        put(path, {"region": region, "model_call": {"stage": "describe", "input_tokens": 10, "output_tokens": 4, "seconds": 2}})
    put(directory / "description_foreign.json", {"model_call": {"input_tokens": 9999}})
    inventory = {"w1": {"observation_cache": str(directory)}, "alias": {"observation_cache": str(directory)}}
    for index in (0, 1):
        put(variant / f"bcnb_1000_sample/full/shard_{index}/image_inventory.json", inventory)
    value = report.build_report(root, bootstrap=10)[-1]["bcnb_1000_sample"]["O_ONLY"]["generic_cache_generation"]
    assert value["unique_cache_files"] == 2 and value["cache_directories"] == 1
    assert value["usage"]["raw_calls"] == 2 and value["usage"]["input_tokens"] == 20


def test_invalid_without_process_recovers_decisions_but_not_actions():
    row = sample_rows()[0]
    result = {"status": "invalid", "pred_answer": "invalid", "model_calls": [
        {"stage": "predict", "raw_response": "broken", "input_tokens": 3},
        {"stage": "predict", "raw_response": '{"answer":"A","thinking_steps":"x"}', "input_tokens": 3},
        {"stage": "reflect", "raw_response": '{"sufficient":"Yes"}', "input_tokens": 3},
    ]}
    metrics = report.trajectory_metrics([row], {"results": {row["sample_id"]: result}, "profiles": []})
    assert metrics["first_predict_to_final"]["correct_wrong"] == 1
    assert metrics["counts"]["first_reflect_yes_final_wrong"] == 1
    assert metrics["qa_call_usage"]["raw_calls"] == 3
    assert metrics["actions"] == {}
    assert metrics["counts"]["process_unavailable"] == 1


def test_cli_is_dependency_light_and_produces_incomplete_report(campaign):
    root, _, _, _ = campaign
    completed = subprocess.run([sys.executable, str(Path(report.__file__).resolve()), "--root", str(root), "--bootstrap", "20"],
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["status"] == "incomplete_not_final"
    text = (root / "report.zh.md").read_text()
    assert "不是最终报告" in text and "不自动迭代" in text
    # Module import does not load any model package or torch.
    command = "import sys; from slurm import report_rule_ablations; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)


def test_completed_external_metrics_mismatch_is_rejected(campaign):
    root, samples, _, variant = campaign
    make_arm(variant, samples, {row["sample_id"]: row["ground_truth_label"] for row in samples}, name="O_ONLY", split=True)
    put(variant / "bcnb_1000_sample/metrics.json", {"expected": 4, "received": 4, "missing": 0, "complete": True,
                                                   "closed": {"closed_overall": {"correct": 0}}})
    with pytest.raises(ValueError, match="metrics.json disagrees"):
        report.build_report(root, bootstrap=10)


def test_dataset_specific_plan_identity_and_local_timestamp(campaign):
    root, _, _, variant = campaign
    lock = report.read_json(variant / "bcnb_1000_sample/full/shard_0/run_lock.json")
    plan = report.read_json(root / "plan.json")
    entry = plan["variants"]["O_ONLY"]
    entry["source_hash"] = lock["code_hash"]
    entry["prompt_manifest_hash"] = lock["prompt_manifest_hash"]
    entry["datasets"] = {"bcnb_1000_sample": {"config_hash": report.digest(lock["config"])}}
    put(root / "plan.json", plan)
    value = report.build_report(root, bootstrap=10)[0]
    assert value["timezone"] == "America/New_York"
    assert value["generated_at"].endswith(("-04:00", "-05:00"))
    entry["datasets"]["bcnb_1000_sample"]["config_hash"] = "other"
    put(root / "plan.json", plan)
    with pytest.raises(ValueError, match="config_hash"):
        report.build_report(root, bootstrap=10)


def test_zoom_distribution_and_task_specific_trajectory(campaign):
    root, _, _, variant = campaign
    path = variant / "bcnb_1000_sample/full/shard_0/results/bcnb_0.json"
    value = report.read_json(path)
    value["process"][0].update(action="zoom", zoom_finding={"region": {"scale": 20.0}})
    value["process"][0]["decision"].update(sufficient="No", zoom_level=20)
    put(path, value)
    metrics = report.build_report(root, bootstrap=10)[-1]["bcnb_1000_sample"]["O_ONLY"]
    assert metrics["zoom_executed_scale"] == {"20": 1}
    assert metrics["zoom_requested_scale"] == {"20": 1}
    assert metrics["by_task"]["Tumor"]["actions"]["zoom"] == 1
    assert metrics["by_task"]["Histological grading"]["actions"].get("zoom", 0) == 0
