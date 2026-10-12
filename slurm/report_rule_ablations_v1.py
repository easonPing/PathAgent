"""Four-rule reporting with frozen initial inputs, post-Zoom reviews and family CIs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import report_rule_ablations as legacy

FAMILY = ("A_HE_PRED", "B_CONFLICT", "C_ZOOM_LOOP", "D_SCALE", "ALL4")
_LEGACY_GENERIC = legacy.generic_usage
CALL_FIELDS = ("stage", "seed", "generation", "enable_thinking", "pixel_sha256", "region_metadata",
               "input_size", "processed_sizes", "image_grid_thw", "messages", "prompt", "raw_response",
               "answer_content", "input_tokens", "output_tokens", "at_token_cap", "finish_reason")
EVENT_METADATA = {"current_parent_regions", "active_zoom_regions", "requested_action",
                  "actual_action", "action_executed", "zoom_unvisited_candidate_count"}


def paired_bootstrap(samples, baseline, results, iterations=10000, seed=128):
    """Same paired WSI draws provide ordinary 95% and five-comparison 99% CIs."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    output = {"iterations": iterations, "seed": seed, "method": "paired_WSI_cluster_percentile_bootstrap",
              "delta_accuracy_percent": None, "delta_ci95_percent": None, "delta_ci99_percent": None,
              "family_size": 5, "family_alpha": .05,
              "limitations": "99% intervals use Bonferroni for five prespecified overall comparisons; approximate bootstrap coverage, no rerun variance. Task intervals are exploratory."}
    if any(row["sample_id"] not in baseline or row["sample_id"] not in results for row in samples):
        return {**output, "status": "withheld_incomplete"}
    import numpy as np
    clusters = defaultdict(lambda: [0, 0, 0])
    for sample in samples:
        values = clusters[str(sample["image_id"])]
        values[0] += 1
        values[1] += int(legacy.outcome(sample, baseline[sample["sample_id"]])["correct"])
        values[2] += int(legacy.outcome(sample, results[sample["sample_id"]])["correct"])
    data = np.asarray([clusters[key] for key in sorted(clusters)], dtype=float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, iterations, 256):
        indices = rng.integers(0, len(data), size=(min(256, iterations - start), len(data)))
        totals = data[indices].sum(axis=1)
        draws.append(100 * (totals[:, 2] - totals[:, 1]) / totals[:, 0])
    values = np.concatenate(draws)
    output.update(status="complete", clusters=len(data), questions=len(samples),
                  delta_accuracy_percent=float(100 * (data[:, 2].sum() - data[:, 1].sum()) / data[:, 0].sum()),
                  delta_ci95_percent=np.quantile(values, [.025, .975]).tolist(),
                  delta_ci99_percent=np.quantile(values, [.005, .995]).tolist())
    return output


def frozen_generic_usage(arm, memo, skip=False):
    if skip or not arm["inventories"]:
        return _LEGACY_GENERIC(arm, memo, skip)
    directories = {str(legacy.resolve(info["observation_cache"], shard))
                   for inventory, _, _, shard in arm["inventories"] for info in inventory.values()
                   if info.get("observation_cache")}
    sidecars = [Path(path) / "source_provenance.json" for path in sorted(directories)]
    if not sidecars or not any(path.exists() for path in sidecars):
        return _LEGACY_GENERIC(arm, memo, False)
    if not all(path.exists() for path in sidecars):
        raise ValueError("Mixed frozen and regenerated initial-input provenance within an arm")
    refs = []
    for sidecar in sidecars:
        directory = sidecar.parent
        if not (directory / "observations.json").is_file() or not (directory / "regions.json").is_file():
            raise ValueError("Incomplete frozen initial-input snapshot: " + str(directory))
        refs.append({"path": str(sidecar), "content_hash": legacy.digest(legacy.read_json(sidecar))})
    # A fresh generic call would be logged outside QA; the frozen reader must not generate it.
    return {"status": "frozen_initial_inputs", "usage": legacy.empty_usage(),
            "fresh_generic_calls": 0, "cache_directories": len(directories), "source_provenance": refs,
            "historical_generic_cost_included": False,
            "note": "Shared frozen Generic descriptions and PLIP features were read from observations.json. No new generic generation is charged; historical calls remain at their source_provenance paths. Executor prompt hashes do not rekey this snapshot."}


def post_zoom_metrics(samples, results):
    indexed = {sample["sample_id"]: sample for sample in samples}
    counts, actions, requested_scales, executed_scales = Counter(), Counter(), Counter(), Counter()
    before_after, before_final = Counter(), Counter()
    missing_ids, review_ids, missing_exposure_ids = [], [], []
    for sid, result in results.items():
        sample = indexed[sid]
        process = result.get("process") or []
        calls = result.get("model_calls") or []
        if not process:
            missing_ids.append(sid)
        if any(call.get("stage") == "missing_info" for call in calls):
            missing_exposure_ids.append(sid)
        counts["raw_missing_calls"] += sum(call.get("stage") == "missing_info" for call in calls)
        observed_review, prior_answer = False, None
        for event in process:
            decision = event.get("decision") or {}
            action = event.get("actual_action", event.get("action", "unknown"))
            executed = event.get("action_executed", event.get("action") in ("zoom", "explore"))
            if executed:
                actions[str(action)] += 1
            if event.get("action") == "zoom":
                requested_scales[legacy.scale_key(decision.get("zoom_level", decision.get("recommended_zoom_level")))] += 1
                if executed and event.get("zoom_finding"):
                    executed_scales[legacy.scale_key(event["zoom_finding"]["region"].get("scale"))] += 1
            if event.get("review_source") == "zoom" or bool(event.get("active_zoom_regions")):
                observed_review = True
                counts["review_events"] += 1
                sufficient = str(decision.get("sufficient", "unknown")).lower()
                counts["reflect_" + sufficient] += 1
                if sufficient == "no":
                    counts["rejected_reviews"] += 1
                if executed:
                    counts["post_review_executed_" + str(action)] += 1
                if prior_answer is not None:
                    previous = legacy.answer_label(prior_answer, sample["choices"])
                    following = legacy.answer_label(decision.get("answer"), sample["choices"])
                    gold = sample["ground_truth_label"]
                    before_after[legacy.transition(previous == gold, following == gold)] += 1
                    before_final[legacy.transition(previous == gold, legacy.outcome(sample, result)["correct"])] += 1
            if "answer" in decision:
                prior_answer = decision["answer"]
        if observed_review:
            review_ids.append(sid)
        zoom_seen = False
        for call in calls:
            if call.get("stage") == "describe" and (call.get("region", {}).get("parent_id") is not None
                    or "/z" in str(call.get("region_metadata", ""))):
                zoom_seen = True
            elif call.get("stage") == "describe":
                zoom_seen = False
            elif zoom_seen and call.get("stage") in ("predict", "reflect"):
                counts["raw_post_zoom_" + call["stage"] + "_calls"] += 1
    counts["questions_reviewed"] = len(review_ids)
    counts["questions_with_missing"] = len(missing_exposure_ids)
    return {"counts": dict(counts), "executed_actions": dict(actions),
            "requested_zoom_scale": dict(requested_scales), "executed_zoom_scale": dict(executed_scales),
            "review_sample_ids": sorted(review_ids), "missing_exposure_sample_ids": sorted(missing_exposure_ids),
            "process_unavailable_sample_ids": sorted(missing_ids),
            "before_to_after_review": {key: before_after[key] for key in legacy.TRANSITIONS},
            "before_review_to_final": {key: before_final[key] for key in legacy.TRANSITIONS},
            "note": "Review counts use explicit active_zoom_regions (or review_source=zoom) events. Raw-call recovery is separate; retries are raw calls. Exposure subgroups are descriptive, not randomized scale comparisons."}


def normalize_process(result):
    return [{key: value for key, value in event.items() if key not in EVENT_METADATA}
            for event in result.get("process") or []]


def no_zoom_invariance(samples, baseline, candidate, expected=954):
    if any(sample["sample_id"] not in baseline for sample in samples):
        return {"status": "baseline_incomplete", "expected_count": expected, "passed": None}
    subset = [sample["sample_id"] for sample in samples
              if not any(event.get("zoom_finding") for event in baseline[sample["sample_id"]].get("process") or [])]
    mismatches, missing = [], []
    for sid in subset:
        if sid not in candidate:
            missing.append(sid)
            continue
        old, new = baseline[sid], candidate[sid]
        failed = []
        for field in ("pred_answer", "explanation", "status", "stop_reason", "final_response",
                      "initial_region_count", "total_examined_regions"):
            if old.get(field) != new.get(field):
                failed.append(field)
        if normalize_process(old) != normalize_process(new):
            failed.append("process")
        old_calls = [{key: value for key, value in call.items() if key != "seconds"} for call in old.get("model_calls") or []]
        new_calls = [{key: value for key, value in call.items() if key != "seconds"} for call in new.get("model_calls") or []]
        if old_calls != new_calls:
            failed.append("stable_model_calls")
        if failed:
            mismatches.append({"sample_id": sid, "failed_fields": failed})
    population_ok = expected is None or len(subset) == expected
    passed = population_ok and not mismatches and not missing
    return {"status": "passed" if passed else "incomplete" if missing and not mismatches and population_ok else "failed",
            "passed": passed if not missing else False if mismatches or not population_ok else None,
            "expected_count": expected, "baseline_no_zoom_count": len(subset), "population_matches": population_ok,
            "compared": len(subset) - len(missing), "missing_sample_ids": missing, "mismatches": mismatches,
            "ignored_event_metadata": sorted(EVENT_METADATA), "stable_call_fields": "all keys except seconds",
            "scope": "C_ZOOM_LOOP must preserve full stable outputs and trajectories for baseline cases without an acquired Zoom finding; elapsed seconds and run/source/cache identities are excluded."}


def expected_identity(plan, entry, dataset):
    expected = dict(entry)
    expected.update(entry.get("datasets", {}).get(dataset, {}))
    expected.setdefault("source_hash", plan.get("source_hash"))
    expected.setdefault("prompt_manifest_hash", expected.get("prompt_hash"))
    return expected


def control_metrics(root, plan, dataset):
    output = {}
    spec = plan["datasets"][dataset]
    path = spec.get("validation_manifest") or spec.get("smoke_manifest")
    if not path:
        return output
    samples = legacy.load_manifest(legacy.resolve(path, root))["samples"]
    for name, entry in plan.get("controls", {}).items():
        directory = legacy.resolve(entry["root"], root) / dataset / "validation"
        results = {}
        lock_path = directory / "run_lock.json"
        if lock_path.exists():
            lock = legacy.read_json(lock_path)
            if legacy.digest({key: value for key, value in lock.items() if key != "run_hash"}) != lock.get("run_hash"):
                raise ValueError("Control run_lock mismatch")
            expected = expected_identity(plan, entry, dataset)
            for key, value in (("source_hash", lock["code_hash"]), ("config_hash", legacy.digest(lock["config"])),
                               ("prompt_manifest_hash", lock["prompt_manifest_hash"])):
                if expected.get(key) and expected[key] != value:
                    raise ValueError("Control plan identity mismatch: " + key)
            if lock["samples_hash"] != legacy.digest(samples):
                raise ValueError("Control validation manifest mismatch")
            for result_path in sorted((directory / "results").glob("*.json")):
                row = legacy.read_json(result_path)
                if row.get("run_hash") != lock["run_hash"] or row["sample_id"] in results:
                    raise ValueError("Control result identity/duplicate mismatch")
                results[row["sample_id"]] = row
        elif list((directory / "results").glob("*.json")):
            raise ValueError("Control results have no run lock")
        if set(results) - {sample["sample_id"] for sample in samples}:
            raise ValueError("Unexpected control sample")
        output[name] = {"purpose": "validation_only_not_full_arm", "root": str(directory),
                        "metrics": legacy.group_metrics(samples, results)}
    return output


def build_report(root, bootstrap=10000, seed=128, include_baseline_generic=False):
    root = Path(root).resolve()
    plan = legacy.read_json(root / "plan.json")
    if "BASE_COMPAT" in plan["variants"]:
        raise ValueError("BASE_COMPAT belongs in controls, not full variants")
    if set(plan["variants"]) - set(FAMILY):
        raise ValueError("Unexpected v1 comparison arm")
    with patch.object(legacy, "paired_bootstrap", paired_bootstrap), patch.object(legacy, "generic_usage", frozen_generic_usage):
        values = legacy.build_report(root, bootstrap, seed, include_baseline_generic)
    report, summary, transitions, bad, trajectories = values
    report["report_version"] = "rules-v1"
    report["comparison_family"] = {"members": list(FAMILY), "size": 5, "endpoint": "overall_accuracy_difference_vs_frozen_baseline",
                                    "confidence": .99, "method": "Bonferroni percentile WSI bootstrap",
                                    "secondary_endpoints": "Task/class/balanced-accuracy analyses are exploratory."}
    invariants = plan.get("invariants", {})
    loop_name = invariants.get("zoom_review_arm", "C_ZOOM_LOOP")
    validation_failed = False
    for dataset, spec in plan["datasets"].items():
        samples = legacy.load_manifest(legacy.resolve(spec["full_manifest"], root))["samples"]
        baseline_arm = legacy.load_arm(legacy.resolve(plan["baseline_root"], root), dataset, samples)
        baseline = baseline_arm["results"]
        report["datasets"][dataset]["controls"] = control_metrics(root, plan, dataset)
        for name, entry in plan["variants"].items():
            arm = legacy.load_arm(legacy.resolve(entry["root"], root), dataset, samples, expected_identity(plan, entry, dataset))
            extra = post_zoom_metrics(samples, arm["results"])
            trajectories[dataset][name]["post_zoom_and_exposure"] = extra
            report["datasets"][dataset]["arms"][name]["post_zoom_and_exposure"] = extra
            if name == loop_name:
                expected = invariants.get("baseline_no_zoom_expected", 954 if len(samples) == 1000 else None)
                audit = no_zoom_invariance(samples, baseline, arm["results"], expected)
                report["datasets"][dataset]["no_zoom_invariance"] = audit
                validation_failed |= audit.get("passed") is False
        for row in summary:
            if row["dataset"] != dataset or row["variant"] == legacy.BASELINE:
                continue
            ci = report["datasets"][dataset]["arms"][row["variant"]]["paired_with_baseline"][row["group"]]["bootstrap"]
            interval = ci.get("delta_ci99_percent")
            row["delta_ci99_low"] = interval[0] if interval else None
            row["delta_ci99_high"] = interval[1] if interval else None
            row["familywise_primary_comparison"] = row["group"] == "overall"
    report["scientific_invariants_passed"] = not validation_failed
    if validation_failed:
        report["is_final"] = False
        report["status"] = "complete_with_invariant_failure" if report["complete"] else "incomplete_with_invariant_failure"
    return values


def write_report(root, *values):
    report, summary, transitions, bad, trajectories = values
    legacy.write_report(root, *values)
    fields = list(summary[0]) if summary else []
    for field in ("delta_ci99_low", "delta_ci99_high", "familywise_primary_comparison"):
        if field not in fields:
            fields.append(field)
    legacy.atomic_text(Path(root) / "summary.csv", legacy.csv_text(summary, fields))
    path = Path(root) / "report.zh.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("区间未作多重比较校正，不包含模型重跑方差。", "95%区间未校正；五个预设overall比较另提供99%区间，以Bonferroni控制家族错误率（bootstrap覆盖近似）。分任务结果为探索性；不包含模型重跑方差。")
    text = text.replace("generic 成本从本组 observation key 对应的唯一 description 缓存记录读取，按路径去重，不把同一 WSI 在多个 QA 中的复用重复计费。缓存中的生成调用不代表本次运行重新执行，跨组也可能复用。", "新组读取共享冻结初始Generic/PLIP输入，fresh generic calls=0；历史生成来源记录在source_provenance中，不计作本轮新调用。旧基线Generic可另行读取。")
    text += "\n五个预设主要比较的99%区间（百分点）：\n\n| 组 | Δ accuracy | 99% CI |\n| --- | --- | --- |\n"
    for row in summary:
        if row["group"] == "overall" and row["variant"] != legacy.BASELINE:
            interval = "未齐全" if row.get("delta_ci99_low") is None else f"[{row['delta_ci99_low']:+.2f}, {row['delta_ci99_high']:+.2f}]"
            delta = "—" if row["delta_accuracy_percent"] is None else f"{row['delta_accuracy_percent']:+.2f}"
            text += f"| {row['variant']} | {delta} | {interval} |\n"
    for dataset, item in report["datasets"].items():
        audit = item.get("no_zoom_invariance")
        if audit:
            text += f"\n{dataset} C组无Zoom不变性：{audit['status']}；已比较{audit.get('compared', 0)}题，差异{len(audit.get('mismatches', []))}题。\n"
    text += "\npost-Zoom审查与倍率决策暴露统计单列在trajectory_metrics.json；暴露子集不作随机因果解释。BASE_COMPAT只报告validation，不进入1000题正式比较。\n"
    if not report["scientific_invariants_passed"]:
        text = "**不变性核验失败：结果已保存，但不构成通过验收的最终比较。**\n\n" + text
    legacy.atomic_text(path, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--include-baseline-generic", action="store_true")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    values = build_report(args.root, args.bootstrap, args.seed, args.include_baseline_generic)
    write_report(args.root, *values)
    print(json.dumps({"report": str(Path(args.root).resolve() / "report.json"), "status": values[0]["status"]}))
    if not values[0]["scientific_invariants_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
