"""Read-only scoring of frozen rule arms; writes reports, never submits or promotes.

Only the standard library and NumPy are used. Missing results retain their manifest
denominator but are excluded from paired transitions and regression examples.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


BASELINE = "BASELINE"
TRANSITIONS = ("correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def resolve(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def normalized(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def answer_label(answer, choices):
    """Match the repository's strict closed-choice scorer, without model imports."""
    if not choices or answer is None:
        return None
    text = str(answer).strip()
    match = re.fullmatch(r"\(?([A-Z])\)?[.)]?", text.upper())
    if match and match.group(1) in choices:
        return match.group(1)
    match = re.match(r"^\(?([A-Z])\)?[.)]\s*(.+)$", text, re.S)
    if match:
        label, body = match.groups()
        return label if label in choices and normalized(body) == normalized(choices[label]) else None
    matches = [key for key, value in choices.items() if normalized(value) == normalized(text)]
    return matches[0] if len(matches) == 1 else None


def load_manifest(path):
    manifest = read_json(path)
    samples = manifest["samples"]
    identities = [row["sample_id"] for row in samples]
    if not samples or len(identities) != len(set(identities)):
        raise ValueError("Manifest must contain unique, nonempty sample IDs: " + str(path))
    if manifest.get("samples_hash") != digest(samples):
        raise ValueError("Manifest samples_hash mismatch: " + str(path))
    for sample in samples:
        if not sample.get("image_id") or not sample.get("choices"):
            raise ValueError("WSI paired reporting requires image_id and closed choices")
        if sample.get("ground_truth_label") not in sample["choices"]:
            raise ValueError("Invalid manifest ground truth: " + sample["sample_id"])
    return manifest


def outcome(sample, result):
    if result is None:
        return {"state": "missing", "label": None, "correct": False}
    label = answer_label(result.get("pred_answer"), sample["choices"])
    valid = result.get("status") == "ok" and label is not None
    return {"state": "ok" if valid else "invalid", "label": label,
            "correct": bool(valid and label == sample["ground_truth_label"])}


def load_arm(root, dataset, samples, expected=None):
    """Each shard has its own run hash; configuration/source/prompt must agree."""
    base = Path(root) / dataset
    indexed = {row["sample_id"]: row for row in samples}
    results, paths, locks, signatures, inventories, profiles = {}, {}, {}, set(), [], []
    scope_verified, warnings = [], []
    for shard in sorted((base / "full").glob("shard_*")):
        if not shard.is_dir():
            continue
        files = sorted((shard / "results").glob("*.json"))
        lock_path = shard / "run_lock.json"
        if not lock_path.exists():
            if files:
                raise ValueError("Results without run_lock: " + str(shard))
            continue
        lock = read_json(lock_path)
        required = ("config", "code_hash", "samples_hash", "protocol", "prompt_manifest_hash", "run_hash")
        if any(key not in lock for key in required):
            raise ValueError("Incomplete run_lock: " + str(lock_path))
        identity = {key: value for key, value in lock.items() if key != "run_hash"}
        if digest(identity) != lock["run_hash"]:
            raise ValueError("run_lock hash mismatch: " + str(lock_path))
        signature = (digest(lock["config"]), lock["code_hash"], lock["protocol"], lock["prompt_manifest_hash"])
        signatures.add(signature)
        if len(signatures) > 1:
            raise ValueError("Mixed configuration/source/protocol/prompt hashes within arm: " + str(base))
        for field, actual in (("config_hash", signature[0]), ("source_hash", signature[1]),
                              ("prompt_manifest_hash", signature[3])):
            if expected and expected.get(field) and expected[field] != actual:
                raise ValueError("Arm disagrees with plan " + field + ": " + str(base))
        prompt_path = shard / "prompt_manifest.json"
        prompt = read_json(prompt_path) if prompt_path.exists() else None
        if prompt is not None and digest(prompt) != lock["prompt_manifest_hash"]:
            raise ValueError("Prompt manifest hash mismatch: " + str(prompt_path))
        shard_manifest = base / "manifests" / (shard.name + ".json")
        shard_ids = None
        if shard_manifest.exists():
            manifest = load_manifest(shard_manifest)
            if manifest["samples_hash"] != lock["samples_hash"]:
                raise ValueError("Shard manifest/run_lock mismatch: " + str(shard))
            shard_ids = {sample["sample_id"] for sample in manifest["samples"]}
            for sample in manifest["samples"]:
                if sample["sample_id"] not in indexed or sample != indexed[sample["sample_id"]]:
                    raise ValueError("Shard contents differ from full manifest: " + str(shard))
            scope_verified.append(shard.name)
        else:
            warnings.append("Shard manifest unavailable: " + str(shard_manifest))
        locks[str(shard)] = lock
        for path in files:
            result = read_json(path)
            sid = result["sample_id"]
            if sid in results:
                raise ValueError("Duplicate result sample_id: " + sid)
            if sid not in indexed or (shard_ids is not None and sid not in shard_ids):
                raise ValueError("Result outside manifest/shard: " + sid)
            if result.get("run_hash") != lock["run_hash"]:
                raise ValueError("Result/run_lock hash mismatch: " + str(path))
            if result.get("protocol", lock["protocol"]) != lock["protocol"]:
                raise ValueError("Result/run_lock protocol mismatch: " + str(path))
            if str(result.get("image_id", indexed[sid]["image_id"])) != str(indexed[sid]["image_id"]):
                raise ValueError("Result image_id mismatch: " + sid)
            results[sid], paths[sid] = result, str(path.resolve())
        inventory_path = shard / "image_inventory.json"
        if inventory_path.exists():
            inventories.append((read_json(inventory_path), lock, prompt, shard))
        profile_path = shard / "profile.json"
        if profile_path.exists():
            profile = read_json(profile_path)
            if profile.get("run_hash") != lock["run_hash"]:
                raise ValueError("Profile/run_lock mismatch: " + str(profile_path))
            profiles.append(profile)
    provenance = {"root": str(Path(root).resolve()), "results_root": str((base / "full").resolve()),
                  "shards_with_locks": len(locks), "shards_with_verified_manifests": scope_verified,
                  "hashes": dict(zip(("config_hash", "code_hash", "protocol", "prompt_manifest_hash"),
                                      next(iter(signatures)))) if signatures else {}, "warnings": warnings}
    return {"results": results, "paths": paths, "locks": locks, "inventories": inventories,
            "profiles": profiles, "provenance": provenance, "metrics_path": base / "metrics.json"}


def class_key(sample, label):
    # Option letters are shuffled and have different meanings across tasks/cohorts.
    prefix = "/".join(str(sample.get(key) or "") for key in ("task", "source")).rstrip("/")
    return prefix + "::" + sample["choices"][label]


def group_metrics(samples, results):
    support, hits, confusion = Counter(), Counter(), defaultdict(Counter)
    counts = Counter()
    for sample in samples:
        value = outcome(sample, results.get(sample["sample_id"]))
        counts[value["state"]] += 1
        counts["correct"] += value["correct"]
        gold = class_key(sample, sample["ground_truth_label"])
        prediction = class_key(sample, value["label"]) if value["state"] == "ok" else "__" + value["state"] + "__"
        support[gold] += 1
        hits[gold] += value["correct"]
        confusion[gold][prediction] += 1
    recalls = {key: 100 * hits[key] / count for key, count in sorted(support.items())}
    return {"expected": len(samples), "received": len(samples) - counts["missing"],
            "missing": counts["missing"], "invalid": counts["invalid"], "correct": counts["correct"],
            "complete": counts["missing"] == 0,
            "accuracy_percent": 100 * counts["correct"] / len(samples),
            "balanced_accuracy_percent": sum(recalls.values()) / len(recalls),
            "class_support": dict(sorted(support.items())), "class_correct": dict(sorted(hits.items())),
            "class_recall_percent": recalls,
            "class_confusion": {key: dict(sorted(value.items())) for key, value in sorted(confusion.items())}}


def sample_groups(samples):
    groups = {"overall": samples}
    for field in ("task", "source"):
        for name in sorted({sample.get(field) for sample in samples if sample.get(field)}):
            groups[field + ":" + name] = [sample for sample in samples if sample.get(field) == name]
    return groups


def paired_bootstrap(samples, baseline, results, iterations=10000, seed=128):
    """Resample WSIs, carrying every QA on the sampled WSI and its multiplicity."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if any(row["sample_id"] not in baseline or row["sample_id"] not in results for row in samples):
        return {"status": "withheld_incomplete", "iterations": iterations, "seed": seed,
                "delta_accuracy_percent": None, "delta_ci95_percent": None}
    import numpy as np
    clusters = defaultdict(lambda: [0, 0, 0])
    for row in samples:
        values = clusters[str(row["image_id"])]
        values[0] += 1
        values[1] += int(outcome(row, baseline[row["sample_id"]])["correct"])
        values[2] += int(outcome(row, results[row["sample_id"]])["correct"])
    data = np.asarray([clusters[key] for key in sorted(clusters)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, iterations, 256):
        draws = rng.integers(0, len(data), size=(min(256, iterations - start), len(data)))
        totals = data[draws].sum(axis=1)
        estimates.append(100 * (totals[:, 2] - totals[:, 1]) / totals[:, 0])
    interval = np.quantile(np.concatenate(estimates), [0.025, 0.975]).tolist()
    return {"status": "complete", "method": "paired_WSI_cluster_percentile_bootstrap",
            "iterations": iterations, "seed": seed, "clusters": len(data), "questions": len(samples),
            "delta_accuracy_percent": float(100 * (data[:, 2].sum() - data[:, 1].sum()) / data[:, 0].sum()),
            "delta_ci95_percent": interval,
            "limitations": "Unadjusted exploratory interval; does not include model rerun variance or multiple-comparison correction."}


def transition(left, right):
    return ("correct" if left else "wrong") + "_" + ("correct" if right else "wrong")


def parse_object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    # Model outputs may have a code fence or a think block preceding their JSON.
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.S)
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = json.JSONDecoder().raw_decode(value[match.start():])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parsed_call(call):
    for field in ("parsed_response", "parsed", "answer_content", "raw_response"):
        parsed = parse_object(call.get(field))
        if parsed is not None:
            return parsed
    return {}


def accepted_stage_calls(calls, stage, field):
    """Retries remain raw calls; use the last valid response of a retry block."""
    blocks, current = [], []
    for call in calls + [{"stage": "__end__"}]:
        if call.get("stage") == stage:
            parsed = parsed_call(call)
            if field in parsed:
                current.append(parsed)
        elif current:
            blocks.append(current[-1])
            current = []
    return blocks


def prediction_details(sample, result):
    process = result.get("process") or []
    calls = result.get("model_calls") or []
    predictions = [event.get("decision", {}) for event in process if "answer" in event.get("decision", {})]
    if not predictions:
        predictions = accepted_stage_calls(calls, "predict", "answer")
    reflections = [event["decision"]["sufficient"] for event in process
                   if "sufficient" in event.get("decision", {})]
    if not reflections:
        reflections = [row["sufficient"] for row in accepted_stage_calls(calls, "reflect", "sufficient")]
    labels = [answer_label(row.get("answer"), sample["choices"]) for row in predictions]
    return {"first_predict_label": labels[0] if labels else None,
            "last_predict_label": labels[-1] if labels else None,
            "predict_available": bool(predictions), "reflect": [str(value).lower() for value in reflections],
            "process_available": bool(process), "process": process}


def empty_usage():
    return {"raw_calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0,
            "calls_without_input_tokens": 0, "calls_without_output_tokens": 0, "calls_without_seconds": 0,
            "at_token_cap": 0}


def scale_key(value):
    try:
        number = float(value)
        return format(number, "g") if math.isfinite(number) else "unknown"
    except (TypeError, ValueError):
        return "unknown"


def add_call(usage, call):
    usage["raw_calls"] += 1
    usage["at_token_cap"] += bool(call.get("at_token_cap"))
    for key in ("input_tokens", "output_tokens", "seconds"):
        value = call.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            usage[key] += value
        else:
            usage["calls_without_" + key] += 1


def trajectory_metrics(samples, arm):
    indexed = {row["sample_id"]: row for row in samples}
    stages, actions, reflections, zoom_levels, requested_levels, stops = {}, Counter(), Counter(), Counter(), Counter(), Counter()
    first_transitions, last_transitions = Counter(), Counter()
    usage = empty_usage()
    counts = Counter()
    seconds = 0.0
    for sid, result in arm["results"].items():
        sample = indexed[sid]
        final = outcome(sample, result)
        details = prediction_details(sample, result)
        if details["predict_available"]:
            first_transitions[transition(details["first_predict_label"] == sample["ground_truth_label"], final["correct"])] += 1
            last_transitions[transition(details["last_predict_label"] == sample["ground_truth_label"], final["correct"])] += 1
            counts["first_predict_final_label_changes"] += details["first_predict_label"] != final["label"]
            counts["last_predict_final_label_changes"] += details["last_predict_label"] != final["label"]
        else:
            counts["predict_unavailable"] += 1
        counts["process_unavailable"] += not details["process_available"]
        if details["reflect"]:
            counts["first_reflect_yes"] += details["reflect"][0] == "yes"
            counts["first_reflect_yes_final_wrong"] += details["reflect"][0] == "yes" and not final["correct"]
        reflections.update(details["reflect"])
        seen_actions = set()
        for event in details["process"]:
            action = event.get("action", "unknown")
            actions[action] += 1
            seen_actions.add(action)
            if action == "zoom":
                region = (event.get("zoom_finding") or {}).get("region") or {}
                if region:
                    zoom_levels[scale_key(region.get("scale"))] += 1
                else:
                    counts["zoom_without_saved_finding"] += 1
                level = event.get("decision", {}).get("zoom_level", event.get("decision", {}).get("recommended_zoom_level"))
                requested_levels[scale_key(level)] += 1
        for action in seen_actions:
            counts["questions_with_" + action] += 1
        stops[str(result.get("stop_reason", "unavailable"))] += 1
        if isinstance(result.get("agent_seconds"), (int, float)):
            seconds += result["agent_seconds"]
        else:
            counts["questions_without_agent_seconds"] += 1
        calls = result.get("model_calls") or []
        counts["questions_without_model_calls"] += not bool(calls)
        for call in calls:
            stage = str(call.get("stage", "unknown"))
            add_call(usage, call)
            add_call(stages.setdefault(stage, empty_usage()), call)
    return {"received_questions": len(arm["results"]), "qa_call_usage": usage, "qa_stages": stages,
            "agent_seconds_sum": seconds, "profile_seconds_sum": sum(profile.get("seconds", 0) for profile in arm["profiles"]),
            "profile_count": len(arm["profiles"]), "counts": dict(counts), "actions": dict(actions),
            "reflect_decisions": dict(reflections), "zoom_executed_scale": dict(zoom_levels),
            "zoom_requested_scale": dict(requested_levels), "stop_reasons": dict(stops),
            "first_predict_to_final": {key: first_transitions[key] for key in TRANSITIONS},
            "last_predict_to_final": {key: last_transitions[key] for key in TRANSITIONS},
            "timing_note": "Call/agent/profile seconds overlap; do not add them. Summed shard seconds are not campaign wall time. Missing process actions are not reconstructed."}


def generic_usage(arm, memo, skip=False):
    output = {"status": "skipped" if skip else "recorded_cache_generation", "usage": empty_usage(),
              "unique_cache_files": 0, "cache_directories": 0, "missing_files": 0,
              "unresolved_inventories": 0, "warnings": [],
              "note": "Unique persisted generic generation calls, deduplicated across QA/shards. Cache reuse does not imply a new call or runtime charge in this arm; costs may overlap other arms."}
    if skip:
        output["note"] += " Historical baseline caches omitted; use --include-baseline-generic to inspect them."
        return output
    if arm["results"] and not arm["inventories"]:
        output["unresolved_inventories"] += 1
        output["warnings"].append("Received QA results have no image_inventory provenance.")
    seen, directories = set(), set()
    for inventory, lock, prompt, shard in arm["inventories"]:
        if prompt is None:
            output["unresolved_inventories"] += 1
            continue
        config = lock["config"]
        if any(key not in config for key in ("models", "generation", "seed")):
            output["unresolved_inventories"] += 1
            continue
        key = digest({"models": config["models"], "generation": config["generation"],
                      "seed": config["seed"], "protocol": prompt})
        for info in inventory.values():
            if not info.get("observation_cache"):
                output["unresolved_inventories"] += 1
                continue
            directory = resolve(info["observation_cache"], shard)
            directories.add(str(directory))
            regions_path = directory / "regions.json"
            if not regions_path.exists():
                output["unresolved_inventories"] += 1
                continue
            for region in read_json(regions_path).get("regions", []):
                path = directory / ("description_" + digest([key, region["region_id"]]) + ".json")
                if path in seen:
                    continue
                seen.add(path)
                if not path.exists():
                    output["missing_files"] += 1
                    continue
                if str(path) not in memo:
                    record = read_json(path)
                    if record.get("region", {}).get("region_id") != region["region_id"]:
                        raise ValueError("Generic cache region mismatch: " + str(path))
                    memo[str(path)] = {key: value for key, value in record.get("model_call", {}).items()
                                       if key in ("stage", "input_tokens", "output_tokens", "seconds", "at_token_cap")}
                call = memo[str(path)]
                if not call:
                    output["warnings"].append("Generic cache has no model_call: " + str(path))
                    continue
                add_call(output["usage"], call)
                output["unique_cache_files"] += 1
    output["cache_directories"] = len(directories)
    if output["missing_files"] or output["unresolved_inventories"] or output["warnings"]:
        output["status"] = "partial_cache_provenance"
    return output


def evidence_candidates(sample, result):
    """High-recall text screening only; no pixel truth or clinical validation."""
    candidates = []
    explanation = str(result.get("explanation") or "")
    def add(tag, text, location):
        if not any(row["candidate_tag"] == tag for row in candidates):
            candidates.append({"candidate_tag": tag, "source": location, "excerpt": text[:500]})
    if re.search(r"(?:cannot|unable to|not provided|unavailable|absence of.{0,35}status)", explanation, re.I) and re.search(
            r"(?:negative|triple.negative|equivocal|2\+)", explanation, re.I):
        add("unknown_to_label_candidate", explanation, "explanation")
    texts = [(explanation, "explanation", None)]
    for index, event in enumerate(result.get("process") or []):
        for number, finding in enumerate(event.get("findings") or []):
            texts.append((str(finding.get("description") or ""),
                          f"process[{index}].findings[{number}].description", finding.get("region") or {}))
        if event.get("zoom_finding"):
            finding = event["zoom_finding"]
            texts.append((str(finding.get("description") or ""),
                          f"process[{index}].zoom_finding.description", finding.get("region") or {}))
    for text, location, region in texts:
        assay = re.search(r"(?:ER|PR|HER2|Ki.?67|p53|S100|HMB45|Melan.A)\b.{0,50}(?:positive|negative|3\+|amplifi)|(?:strong|complete|circumferential).{0,35}(?:membranous|membrane) staining", text, re.I)
        if assay:
            add("assay_assertion_candidate", text[max(0, assay.start() - 60):assay.end() + 180], location)
        if region and region.get("magnification_source") == "assumed" and float(region.get("scale", 0) or 0) <= 5:
            count = re.search(r"(?:\d+\s*(?:/|per)\s*(?:10\s*)?HPF|\d+.{0,12}mito.{0,25}HPF|40\s*[×x])", text, re.I)
            if count:
                add("low_scale_hpf_claim_candidate", text[max(0, count.start() - 60):count.end() + 180], location)
        if region and re.search(r"(?:no evidence of malignancy|no invasive growth|benign lesion|fibroadenoma)", text, re.I) and re.search(r"(?:invasive ductal carcinoma|infiltrat\w+ carcinoma|confirmed malignancy)", text, re.I):
            add("same_region_conflict_candidate", text, location)
        if sample.get("dataset") == "bcnb" and re.search(r"Wisconsin|WHO/ISUP|renal cell carcinoma", text, re.I):
            add("organ_or_grading_system_candidate", text, location)
    details = prediction_details(sample, result)
    final = outcome(sample, result)
    if details["predict_available"] and details["last_predict_label"] != final["label"]:
        calls = result.get("model_calls") or []
        positions = [index for index, call in enumerate(calls) if call.get("stage") == "predict"]
        if positions and not any(call.get("stage") == "describe" for call in calls[positions[-1] + 1:]):
            add("final_changed_without_new_description_candidate", explanation, "model_calls/predict_to_final")
    return candidates


def compact_trajectory(result):
    return [{"iteration": event.get("iteration"), "action": event.get("action"),
             "decision": event.get("decision"),
             "regions": [{key: finding.get("region", {}).get(key) for key in ("region_id", "scale", "scale_kind", "magnification_source")}
                         for finding in event.get("findings") or []],
             "zoom_region": (event.get("zoom_finding") or {}).get("region"),
             "zoom_description": (event.get("zoom_finding") or {}).get("description")}
            for event in result.get("process") or []]


def reconcile_metrics(arm, computed):
    path = arm["metrics_path"]
    if not path.exists():
        return {"status": "not_available", "path": str(path)}
    existing = read_json(path)
    old = existing.get("closed", {}).get("closed_overall", {})
    checks = {"expected": existing.get("expected"), "received": existing.get("received"),
              "missing": existing.get("missing"), "correct": old.get("correct")}
    if existing.get("complete") and computed["complete"] and any(
            value is not None and value != computed[key] for key, value in checks.items()):
        raise ValueError("Completed metrics.json disagrees with manifest scoring: " + str(path))
    matching = all(value is None or value == computed[key] for key, value in checks.items())
    return {"status": "consistent" if matching else "stale_or_partial", "path": str(path),
            "note": "Report reparses strict choices; status=ok with an illegal option is reported invalid."}


def csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_report(root, bootstrap=10000, seed=128, include_baseline_generic=False):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    baseline_root = resolve(plan["baseline_root"], root)
    variants = plan["variants"]
    if BASELINE in variants or not isinstance(variants, dict):
        raise ValueError("variants must be a mapping without the reserved BASELINE name")
    roots = [baseline_root] + [resolve(spec["root"], root) for spec in variants.values()]
    if len(roots) != len(set(roots)):
        raise ValueError("Baseline and variant roots must be distinct; arms are never merged")
    now = datetime.now(timezone.utc)
    report = {"schema_version": 1, "generated_at": now.astimezone(ZoneInfo("America/New_York")).isoformat(),
              "generated_at_utc": now.isoformat(), "timezone": "America/New_York",
              "campaign_root": str(root), "plan_hash": digest(plan), "complete": True, "is_final": False,
              "bootstrap": {"iterations": bootstrap, "seed": seed, "unit": "WSI"}, "datasets": {},
              "candidate_warning": "Automated text-screening candidate tags are not confirmed errors or pixel truth.",
              "policy": "No external judge, automatic rule iteration, selection, promotion, or submission."}
    summary_rows, transition_rows, bad_rows, trajectories, generic_memo = [], [], [], {}, {}
    for dataset, spec in plan["datasets"].items():
        manifest_path = resolve(spec["full_manifest"], root)
        manifest = load_manifest(manifest_path)
        samples = manifest["samples"]
        groups = sample_groups(samples)
        arms = {BASELINE: load_arm(baseline_root, dataset, samples)}
        for variant, entry in variants.items():
            expected = dict(entry)
            expected.update(entry.get("datasets", {}).get(dataset, {}))
            expected.setdefault("source_hash", plan.get("source_hash"))
            expected.setdefault("prompt_manifest_hash", expected.get("prompt_hash"))
            arms[variant] = load_arm(resolve(entry["root"], root), dataset, samples, expected)
        baseline = arms[BASELINE]["results"]
        baseline_metrics = group_metrics(samples, baseline)
        expected_baseline = spec.get("baseline_expected_correct", 513 if dataset == "bcnb_1000_sample" and len(samples) == 1000 else None)
        if expected_baseline is not None and baseline_metrics["complete"] and baseline_metrics["correct"] != expected_baseline:
            raise ValueError(f"Baseline verification failed: expected {expected_baseline}, got {baseline_metrics['correct']}")
        dataset_report = {"manifest": str(manifest_path), "samples_hash": manifest["samples_hash"],
                          "expected": len(samples), "baseline_verification": {
                              "expected_correct": expected_baseline, "observed_correct": baseline_metrics["correct"],
                              "status": "verified" if expected_baseline is not None and baseline_metrics["complete"] else "not_verified"},
                          "arms": {}}
        trajectories[dataset] = {}
        for name, arm in arms.items():
            results = arm["results"]
            metrics = {group: group_metrics(rows, results) for group, rows in groups.items()}
            paired = {}
            complete = metrics["overall"]["complete"]
            report["complete"] &= complete
            for group, rows in groups.items():
                counts = Counter({key: 0 for key in TRANSITIONS})
                paired_count = 0
                for sample in rows:
                    sid = sample["sample_id"]
                    if sid not in baseline or sid not in results:
                        continue
                    paired_count += 1
                    counts[transition(outcome(sample, baseline[sid])["correct"], outcome(sample, results[sid])["correct"])] += 1
                ci = paired_bootstrap(rows, baseline, results, bootstrap, seed) if name != BASELINE else None
                paired[group] = {"paired_received": paired_count, "unpaired_missing": len(rows) - paired_count,
                                 "transitions": dict(counts), "bootstrap": ci}
                metric = metrics[group]
                interval = (ci or {}).get("delta_ci95_percent")
                summary_rows.append({"dataset": dataset, "variant": name, "group": group,
                                     **{key: metric[key] for key in ("expected", "received", "missing", "invalid", "correct", "accuracy_percent", "balanced_accuracy_percent", "complete")},
                                     "paired_received": paired_count, **dict(counts),
                                     "delta_accuracy_percent": (ci or {}).get("delta_accuracy_percent"),
                                     "delta_ci95_low": interval[0] if interval else None,
                                     "delta_ci95_high": interval[1] if interval else None})
            trajectory = trajectory_metrics(samples, arm)
            trajectory["by_task"] = {
                group[5:]: trajectory_metrics(rows, {
                    "results": {sample["sample_id"]: results[sample["sample_id"]]
                                for sample in rows if sample["sample_id"] in results}, "profiles": []})
                for group, rows in groups.items() if group.startswith("task:")}
            trajectory["generic_cache_generation"] = generic_usage(arm, generic_memo, skip=name == BASELINE and not include_baseline_generic)
            trajectories[dataset][name] = trajectory
            dataset_report["arms"][name] = {"complete": complete, "provenance": arm["provenance"], "metrics": metrics,
                                           "paired_with_baseline": paired, "cost_and_trajectory": trajectory,
                                           "existing_metrics": reconcile_metrics(arm, metrics["overall"])}
            candidate_counts, selected_counts = Counter(), Counter()
            for sample in samples:
                sid = sample["sample_id"]
                result = results.get(sid)
                left = outcome(sample, baseline.get(sid))
                right = outcome(sample, result)
                pair = transition(left["correct"], right["correct"]) if left["state"] != "missing" and right["state"] != "missing" else None
                details = prediction_details(sample, result) if result else {}
                record = {"dataset": dataset, "variant": name, "sample_id": sid, "image_id": sample["image_id"],
                          "task": sample.get("task", ""), "source": sample.get("source", ""),
                          "ground_truth_label": sample["ground_truth_label"], "baseline_label": left["label"],
                          "prediction_label": right["label"], "baseline_state": left["state"], "state": right["state"],
                          "baseline_correct": left["correct"] if left["state"] != "missing" else None,
                          "correct": right["correct"] if right["state"] != "missing" else None,
                          "transition": pair, "first_predict_label": details.get("first_predict_label"),
                          "last_predict_label": details.get("last_predict_label"),
                          "first_reflect": next(iter(details.get("reflect", [])), None),
                          "result_path": arm["paths"].get(sid), "baseline_result_path": arms[BASELINE]["paths"].get(sid)}
                transition_rows.append(record)
                if result is None:
                    continue  # Missing work is not a regression or an evidence example.
                candidates = evidence_candidates(sample, result)
                candidate_counts.update(item["candidate_tag"] for item in candidates)
                if not right["correct"] or pair == "correct_wrong" or candidates:
                    reasons = (["wrong"] if not right["correct"] else []) + (["regressed"] if name != BASELINE and pair == "correct_wrong" else [])
                    if right["state"] == "invalid":
                        reasons.append("invalid")
                    if candidates:
                        reasons.append("automated_evidence_candidate")
                    selected_counts.update(reasons)
                    bad_rows.append({**record, "selection_reasons": reasons, "question": sample.get("question"),
                                     "choices": sample["choices"], "pred_answer": result.get("pred_answer"),
                                     "explanation": result.get("explanation"), "candidate_tags": [item["candidate_tag"] for item in candidates],
                                     "candidate_evidence": candidates, "candidate_is_pixel_truth": False,
                                     "trajectory": compact_trajectory(result),
                                     "model_call_count": len(result.get("model_calls") or []),
                                     "process_available": bool(result.get("process"))})
            dataset_report["arms"][name]["bad_subset_counts"] = dict(selected_counts)
            dataset_report["arms"][name]["evidence_candidate_counts"] = dict(candidate_counts)
        report["datasets"][dataset] = dataset_report
    report["is_final"] = report["complete"]
    report["status"] = "complete" if report["complete"] else "incomplete_not_final"
    return report, summary_rows, transition_rows, bad_rows, trajectories


def write_report(root, report, summary_rows, transition_rows, bad_rows, trajectories):
    root = Path(root)
    summary_fields = ("dataset", "variant", "group", "expected", "received", "missing", "invalid", "correct",
                      "accuracy_percent", "balanced_accuracy_percent", "complete", "paired_received", *TRANSITIONS,
                      "delta_accuracy_percent", "delta_ci95_low", "delta_ci95_high")
    transition_fields = ("dataset", "variant", "sample_id", "image_id", "task", "source", "ground_truth_label",
                         "baseline_label", "prediction_label", "baseline_state", "state", "baseline_correct", "correct",
                         "transition", "first_predict_label", "last_predict_label", "first_reflect", "result_path", "baseline_result_path")
    lines = ["# 手工规则消融报告", "", "状态：" + ("结果齐全。" if report["complete"] else "**结果未齐全，不是最终报告。**"), "",
             "评分固定使用 full manifest 分母，invalid 和 missing 不改变分母。缺失结果不记作配对退化案例；不完整组不生成提升置信区间。", "",
             "| 数据集 | 组 | 已收到/分母 | 正确 | accuracy | balanced accuracy | 相对基线 Δ / 95% CI（百分点） |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary_rows:
        if row["group"] != "overall":
            continue
        delta = "—"
        if row["delta_accuracy_percent"] is not None:
            delta = f"{row['delta_accuracy_percent']:+.2f} [{row['delta_ci95_low']:+.2f}, {row['delta_ci95_high']:+.2f}]"
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['received']}/{row['expected']} | {row['correct']} | {row['accuracy_percent']:.2f}% | {row['balanced_accuracy_percent']:.2f}% | {delta} |")
    lines += ["", "balanced accuracy 为 manifest 中任务/来源/语义类别召回的平均，不混合不同含义的选项字母。任务和类别混淆见 report.json、summary.csv。", "",
              f"配对区间按 WSI 整组重采样 {report['bootstrap']['iterations']:,} 次，seed={report['bootstrap']['seed']}；保留一张 WSI 的全部问答和重复抽样权重。区间未作多重比较校正，不包含模型重跑方差。", "",
              "transitions.csv 保留逐题四种正确性转移、missing 状态、Predict→Final 及结果路径。bad_subset.jsonl 包含已收到的所有错题、退化、invalid 和自动文本候选（含部分答对题）。候选标签不是人工确认错误，更不是像素真值；本工具未运行外部 judge。", "",
              "trajectory_metrics.json 分别统计 QA 原始调用（含重试）、token、耗时、Reflect、Explore/Zoom 及倍率。generic 成本从本组 observation key 对应的唯一 description 缓存记录读取，按路径去重，不把同一 WSI 在多个 QA 中的复用重复计费。缓存中的生成调用不代表本次运行重新执行，跨组也可能复用。", "",
              "调用耗时、agent 耗时和 profile 耗时互相包含，不能相加；并行 shard 时间之和不是总墙钟时间。缺少 process 的动作不作猜测补全。默认跳过历史基线 generic 缓存；可使用 --include-baseline-generic 补充。", ""]
    for dataset, value in report["datasets"].items():
        check = value["baseline_verification"]
        lines.append(f"基线核验 {dataset}：正确 {check['observed_correct']}，预期 {check['expected_correct']}，状态 {check['status']}。")
    lines += ["", "本报告不自动迭代、选择或晋升规则，不提交新任务。", ""]
    atomic_text(root / "summary.csv", csv_text(summary_rows, summary_fields))
    atomic_text(root / "transitions.csv", csv_text(transition_rows, transition_fields))
    atomic_text(root / "bad_subset.jsonl", "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in bad_rows))
    atomic_text(root / "trajectory_metrics.json", json_text(trajectories))
    atomic_text(root / "report.zh.md", "\n".join(lines))
    # Publish the report last; consumers can use generated_at/status as a snapshot marker.
    atomic_text(root / "report.json", json_text(report))


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


if __name__ == "__main__":
    main()
