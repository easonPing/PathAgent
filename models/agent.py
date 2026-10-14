"""Paper Algorithm 1; model-independent to allow deterministic control-flow tests."""
import time
from copy import deepcopy
import numpy as np
from data_processing.datasets import public_input
from data_processing.regions import zoom_regions
from models.region_selection import (
    low_region_count, question_region_count, selection_policy, zoom_region_count,
)


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
    """Keep the frozen path unless the explicit Zoom continuation flag is enabled."""
    if config.get('protocol') == 'pathology-rules-v2':
        from models.rules_v2_agent import run_agent_v2
        return run_agent_v2(sample, regions, descriptions, features, backend, config)
    enabled = config["algorithm"].get("zoom_return_to_loop", False)
    if type(enabled) is not bool:
        raise ValueError("zoom_return_to_loop must be a boolean")
    implementation = _run_agent_zoom_loop if enabled else _run_agent_legacy
    return implementation(sample, regions, descriptions, features, backend, config)


def _run_agent_legacy(sample, regions, descriptions, features, backend, config):
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
                       low_region_count(algorithm, n, initial=True))
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
        for name in current[:question_region_count(algorithm, len(current))]:
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
                selected_ids = retrieve([r.region_id for r in candidates], embeddings,
                                        backend.encode_text(missing), zoom_region_count(algorithm, len(candidates)))
                child_by_id = {r.region_id: r for r in candidates}
                acquired = [{"region": child_by_id[name].to_dict(),
                             "description": backend.describe(child_by_id[name], sample["question"], missing_info=missing)}
                            for name in selected_ids]
                evidence.extend(acquired)
                if selection_policy(algorithm) == 'max5':
                    event.update(zoom_findings=acquired, selected_children=selected_ids)
                else:
                    event["zoom_finding"] = acquired[0]
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
                           low_region_count(algorithm, n, initial=False))
        event["next_regions"] = current
    final = backend.conclude(evidence, trajectory, sample)
    return {"pred_answer": final["answer"], "explanation": final["explanation"], "status": "ok",
            "process": trajectory, "stop_reason": stop_reason, "final_response": final,
            "initial_region_count": n, "total_examined_regions": len(visited),
            "agent_seconds": time.monotonic() - started}


def _run_agent_zoom_loop(sample, regions, descriptions, features, backend, config):
    """Share five review rounds across Explore and Zoom, retaining low-scale parents.

    Before the first Zoom the model inputs and legacy history fields are unchanged.
    Acquiring a Zoom child adds a review round without describing its parents again.
    Extra action bookkeeping stays outside ``decision`` so the existing Final
    serializer still receives only the original iteration/action/decision fields.
    """
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
                       low_region_count(algorithm, n, initial=True))
    visited, evidence, trajectory = set(), [], []
    visited_zoom, active_zoom = set(), []
    current_findings = []
    fresh_low_batch = True
    stop_reason = None
    started = time.monotonic()
    for iteration in range(1, algorithm["max_iterations"] + 1):
        if not current:
            stop_reason = "no_unexamined_regions"
            break
        if fresh_low_batch:
            if visited.intersection(current):
                raise RuntimeError("Already examined region selected")
            visited.update(current)
            for name in current[:question_region_count(algorithm, len(current))]:
                descriptions[name] += "\n" + backend.describe(by_id[name], sample["question"])
            current_findings = [{"region": by_id[p].to_dict(), "description": descriptions[p]} for p in current]
            evidence.extend(current_findings)
            fresh_low_batch = False
        findings = current_findings + active_zoom
        kind = regions[0].scale_kind
        allowed = algorithm["zoom_magnifications"] if kind == "physical" else algorithm["relative_zoom_factors"]
        decision = backend.evaluate(findings, sample, kind, allowed)
        event = {"iteration": iteration, "findings": findings, "decision": decision,
                 "examined_region_count": len(visited),
                 "current_parent_regions": list(current),
                 "active_zoom_regions": [f["region"]["region_id"] for f in active_zoom]}
        trajectory.append(event)
        if str(decision.get("sufficient", "")).lower() == "yes":
            event["action"], stop_reason = "conclude", "sufficient"
            event.update(requested_action="conclude", actual_action="conclude", action_executed=True)
            break
        missing = str(decision.get("missing_info") or "").strip()
        if not missing:
            raise ValueError("Missing missing_info")
        if str(decision.get("zoom_recommendation", "")).lower() == "yes":
            event["action"] = "zoom"
            event.update(requested_action="zoom", actual_action=None, action_executed=False)
            target = float(decision["zoom_level"])
            if target not in allowed:
                raise ValueError(f"Unsupported zoom {target}")
            # Every acquired child must have a remaining Predict/Reflect round.
            if iteration == algorithm["max_iterations"]:
                stop_reason = "max_iterations"
                break
            all_candidates = [child for name in current for child in zoom_regions(by_id[name], target)]
            candidates = [child for child in all_candidates if child.region_id not in visited_zoom]
            event.update(zoom_candidate_count=len(all_candidates), zoom_query=missing,
                         zoom_unvisited_candidate_count=len(candidates))
            if not candidates:
                stop_reason = "zoom_has_no_unvisited_children" if all_candidates else "zoom_has_no_valid_children"
                break
            embeddings = backend.encode_regions(candidates)
            selected_ids = retrieve([r.region_id for r in candidates], embeddings,
                                    backend.encode_text(missing), zoom_region_count(algorithm, len(candidates)))
            child_by_id = {r.region_id: r for r in candidates}
            acquired = [{"region": child_by_id[name].to_dict(),
                         "description": backend.describe(child_by_id[name], sample["question"], missing_info=missing)}
                        for name in selected_ids]
            visited_zoom.update(selected_ids)
            active_zoom.extend(acquired)
            evidence.extend(acquired)
            if selection_policy(algorithm) == 'max5':
                event.update(zoom_findings=acquired, selected_children=selected_ids)
            else:
                event["zoom_finding"] = acquired[0]
            event.update(actual_action="zoom", action_executed=True)
            continue
        event["action"] = "explore"
        event.update(requested_action="explore", actual_action=None, action_executed=False)
        if iteration == algorithm["max_iterations"]:
            stop_reason = "max_iterations"
            break
        remaining = [p for p in names if p not in visited]
        if not remaining:
            stop_reason = "no_unexamined_regions"
            break
        current = retrieve(remaining, [features[p] for p in remaining], backend.encode_text(missing),
                           low_region_count(algorithm, n, initial=False))
        event["next_regions"] = current
        event.update(actual_action="explore" if current else None, action_executed=bool(current))
        active_zoom = []
        fresh_low_batch = True
    final = backend.conclude(evidence, trajectory, sample)
    return {"pred_answer": final["answer"], "explanation": final["explanation"], "status": "ok",
            "process": trajectory, "stop_reason": stop_reason, "final_response": final,
            "initial_region_count": n, "total_examined_regions": len(visited),
            "total_zoomed_regions": len(visited_zoom), "total_evidence_regions": len(evidence),
            "agent_seconds": time.monotonic() - started}
