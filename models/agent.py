"""Paper Algorithm 1; model-independent to allow deterministic control-flow tests."""
import math
import time
from copy import deepcopy
import numpy as np
from data_processing.datasets import public_input
from data_processing.regions import zoom_regions


def unit_rows(features):
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Invalid embeddings")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError("Zero embeddings")
    return array / norm


def retrieve(names, embeddings, query_embedding, k):
    if not names:
        return []
    feats, query = unit_rows(embeddings), unit_rows(query_embedding)
    if len(feats) != len(names) or len(query) != 1 or feats.shape[1] != query.shape[1]:
        raise ValueError("Embedding shape or identity mismatch")
    scores = (feats @ query.T).reshape(-1)
    ranked = sorted(range(len(names)), key=lambda i: (-float(scores[i]), names[i]))
    return [names[i] for i in ranked[:min(k, len(names))]]


def run_agent(sample, regions, descriptions, features, backend, config):
    sample = public_input(sample)
    algorithm = config["algorithm"]
    names = [r.region_id for r in regions]
    if not names or len(set(names)) != len(names):
        raise ValueError("Need nonempty, unique regions")
    if set(names) != set(descriptions) or set(names) != set(features):
        raise ValueError("Region/description/feature coverage must match exactly")
    if len({r.scale_kind for r in regions}) != 1 or len({r.scale for r in regions}) != 1:
        raise ValueError("Initial regions must use one scale")
    by_id = {r.region_id: r for r in regions}
    descriptions = deepcopy(descriptions)
    n = len(names)
    current = retrieve(names, [features[p] for p in names], backend.encode_text(sample["question"]),
                       math.ceil(n * algorithm["initial_ratio"]))
    visited, evidence, trajectory = set(), [], []
    stop_reason = None
    started = time.monotonic()
    for iteration in range(1, algorithm["max_iterations"] + 1):
        if not current:
            stop_reason = "no_unexamined_regions"
            break
        if visited.intersection(current):
            raise RuntimeError("Already examined region selected")
        visited.update(current)
        for name in current[:algorithm["question_description_topk"]]:
            descriptions[name] += "\n" + backend.describe(by_id[name], sample["question"])
        findings = [{"region": by_id[p].to_dict(), "description": descriptions[p]} for p in current]
        evidence.extend(findings)
        kind = regions[0].scale_kind
        allowed = algorithm["zoom_magnifications"] if kind == "physical" else algorithm["relative_zoom_factors"]
        decision = backend.evaluate(findings, sample, kind, allowed)
        event = {"iteration": iteration, "findings": findings, "decision": decision,
                 "examined_region_count": len(visited)}
        trajectory.append(event)
        if str(decision.get("sufficient", "")).lower() == "yes":
            event["action"], stop_reason = "conclude", "sufficient"
            break
        missing = str(decision.get("missing_info") or "").strip()
        if not missing:
            raise ValueError("Missing missing_info")
        if str(decision.get("zoom_recommendation", "")).lower() == "yes":
            event["action"] = "zoom"
            target = float(decision["zoom_level"])
            if target not in allowed:
                raise ValueError(f"Unsupported zoom {target}")
            candidates = [child for name in current for child in zoom_regions(by_id[name], target)]
            event.update(zoom_candidate_count=len(candidates), zoom_query=missing)
            if candidates:
                embeddings = backend.encode_regions(candidates)
                selected_id = retrieve([r.region_id for r in candidates], embeddings,
                                       backend.encode_text(missing), 1)[0]
                selected = next(r for r in candidates if r.region_id == selected_id)
                finding = {"region": selected.to_dict(),
                           "description": backend.describe(selected, sample["question"], missing_info=missing)}
                evidence.append(finding)
                event["zoom_finding"] = finding
                stop_reason = "zoom_completed"
            else:
                stop_reason = "zoom_has_no_valid_children"
            break
        event["action"] = "explore"
        if iteration == algorithm["max_iterations"]:
            stop_reason = "max_iterations"
            break
        remaining = [p for p in names if p not in visited]
        if not remaining:
            stop_reason = "no_unexamined_regions"
            break
        current = retrieve(remaining, [features[p] for p in remaining], backend.encode_text(missing),
                           math.ceil(n * algorithm["replenish_ratio"]))
        event["next_regions"] = current
    final = backend.conclude(evidence, trajectory, sample)
    return {"pred_answer": final["answer"], "explanation": final["explanation"], "status": "ok",
            "process": trajectory, "stop_reason": stop_reason, "final_response": final,
            "initial_region_count": n, "total_examined_regions": len(visited),
            "agent_seconds": time.monotonic() - started}
