"""New report statistics must not contaminate the archived v0 report module."""
import copy
import json
from pathlib import Path

import pytest
from slurm import report_rule_ablations_v1 as report
from test_rule_ablation_report import sample_rows, make_arm, manifest, put


def results(rows):
    return {row["sample_id"]: {"pred_answer": row["ground_truth_label"], "explanation": "same", "status": "ok",
        "stop_reason": "sufficient", "process": [{"iteration": 1, "action": "conclude", "decision": {"answer": row["ground_truth_label"], "sufficient": "Yes"}, "findings": []}],
        "model_calls": [{"stage": "final", "seed": 1, "generation": {"do_sample": False}, "seconds": 3, "raw_response": "same"}]}
        for row in rows}


def test_bootstrap_99_is_at_least_as_wide_and_incomplete_withheld():
    rows = sample_rows()
    baseline = results(rows)
    candidate = copy.deepcopy(baseline)
    candidate[rows[0]["sample_id"]]["pred_answer"] = "wrong"
    ci = report.paired_bootstrap(rows, baseline, candidate, 10000, 128)
    assert ci["family_size"] == 5 and ci["delta_accuracy_percent"] == -25
    assert ci["delta_ci99_percent"][0] <= ci["delta_ci95_percent"][0]
    assert ci["delta_ci99_percent"][1] >= ci["delta_ci95_percent"][1]
    candidate.pop(rows[0]["sample_id"])
    assert report.paired_bootstrap(rows, baseline, candidate)["delta_ci99_percent"] is None


def test_invariance_ignores_only_timing_and_bookkeeping():
    rows = sample_rows()
    baseline = results(rows)
    candidate = copy.deepcopy(baseline)
    for value in candidate.values():
        value["model_calls"][0]["seconds"] = 900
        value["process"][0].update(current_parent_regions=[], active_zoom_regions=[], requested_action="conclude", actual_action="conclude", action_executed=True, zoom_unvisited_candidate_count=0)
    assert report.no_zoom_invariance(rows, baseline, candidate, 4)["passed"]
    candidate[rows[0]["sample_id"]]["model_calls"][0]["generation"]["do_sample"] = True
    check = report.no_zoom_invariance(rows, baseline, candidate, 4)
    assert check["status"] == "failed" and len(check["mismatches"]) == 1
    candidate = copy.deepcopy(baseline)
    candidate.pop(rows[0]["sample_id"])
    check = report.no_zoom_invariance(rows, baseline, candidate, 4)
    assert check["status"] == "incomplete" and not check["mismatches"]


def test_post_zoom_metrics_use_executed_actions_and_review_metadata():
    rows = sample_rows()[:1]
    sample = rows[0]
    result = results(rows)[sample["sample_id"]]
    result["process"] = [
        {"decision": {"answer": "B", "sufficient": "No", "zoom_level": 20}, "action": "zoom", "action_executed": True, "actual_action": "zoom", "zoom_finding": {"region": {"scale": 20}}},
        {"decision": {"answer": "A", "sufficient": "No"}, "active_zoom_regions": ["r/z20/0"], "action": "explore", "action_executed": False, "actual_action": None},
    ]
    result["model_calls"] = [{"stage": "describe", "region_metadata": "r/z20/0"}, {"stage": "predict"}, {"stage": "reflect"}]
    value = report.post_zoom_metrics(rows, {sample["sample_id"]: result})
    assert value["counts"]["review_events"] == 1 and value["counts"]["reflect_no"] == 1
    assert value["before_to_after_review"]["wrong_correct"] == 1
    assert value["executed_actions"] == {"zoom": 1}
    assert value["counts"]["raw_post_zoom_predict_calls"] == 1


def test_frozen_generic_does_not_use_new_executor_key(tmp_path):
    directory = tmp_path / "inputs"
    put(directory / "source_provenance.json", {"origin": "/archived/cache"})
    put(directory / "regions.json", {"regions": [{"region_id": "r"}]})
    put(directory / "observations.json", {"descriptions": {"r": "frozen"}, "features": {"r": [1]}})
    arm = {"inventories": [({"image": {"observation_cache": str(directory)}}, {"config": {}}, {"new_executor_prompt": "different"}, tmp_path)], "results": {"test": {}}}
    value = report.frozen_generic_usage(arm, {})
    assert value["status"] == "frozen_initial_inputs" and value["fresh_generic_calls"] == 0
    assert not value["historical_generic_cost_included"]
    assert value["usage"]["raw_calls"] == 0


def test_complete_report_adds_99_csv_and_detects_c_invariance(tmp_path):
    rows = sample_rows()
    baseline = tmp_path / "baseline"
    variant = tmp_path / "C"
    answers = {row["sample_id"]: row["ground_truth_label"] for row in rows}
    make_arm(baseline, rows, answers)
    make_arm(variant, rows, answers, name="C_ZOOM_LOOP")
    manifest(tmp_path / "full.json", rows)
    put(tmp_path / "plan.json", {"baseline_root": str(baseline),
        "datasets": {"bcnb_1000_sample": {"full_manifest": str(tmp_path / "full.json"), "baseline_expected_correct": 4}},
        "variants": {"C_ZOOM_LOOP": {"root": str(variant)}},
        "invariants": {"zoom_review_arm": "C_ZOOM_LOOP", "baseline_no_zoom_expected": 4}})
    original = report.legacy.paired_bootstrap
    values = report.build_report(tmp_path, bootstrap=100)
    assert report.legacy.paired_bootstrap is original
    assert values[0]["scientific_invariants_passed"]
    report.write_report(tmp_path, *values)
    assert "delta_ci99_low" in (tmp_path / "summary.csv").read_text().splitlines()[0]
    assert "99%" in (tmp_path / "report.zh.md").read_text()
    path = next((variant / "bcnb_1000_sample/full").glob("shard_*/results/*.json"))
    value = json.loads(path.read_text())
    value["explanation"] = "unintended change"
    put(path, value)
    value = report.build_report(tmp_path, bootstrap=100)[0]
    assert value["complete"] and not value["is_final"]
    assert value["status"] == "complete_with_invariant_failure"
