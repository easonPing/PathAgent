"""CPU-only control-flow regressions for the opt-in Zoom continuation arm."""
from copy import deepcopy

import pytest

from data_processing.datasets import make_sample
from data_processing.regions import Region, zoom_regions
from models.agent import _run_agent_legacy, run_agent


def decision(action, target=10):
    return {"answer": "A", "thinking_steps": "observed morphology",
            "sufficient": "Yes" if action == "conclude" else "No",
            "missing_info": "nuclear detail", "zoom_recommendation": "Yes" if action == "zoom" else "No",
            "zoom_level": target}


class RecordingBackend:
    def __init__(self, decisions):
        self.decisions = deepcopy(decisions)
        self.calls = []
        self.evaluations = []
        self.candidate_batches = []
        self.described = []
        self.final_evidence = None
        self.final_history = None

    def encode_text(self, text):
        self.calls.append(("encode_text", text))
        return [1., 0.]

    def encode_regions(self, regions):
        serialized = [r.to_dict() for r in regions]
        self.calls.append(("encode_regions", serialized))
        self.candidate_batches.append(serialized)
        return [[1., 0.] for _ in regions]

    def describe(self, region, question, missing_info=None):
        record = (region.to_dict(), question, missing_info)
        self.calls.append(("describe", record))
        self.described.append(record)
        return "observed " + region.region_id

    def evaluate(self, findings, sample, scale_kind, allowed_scales):
        assert not {"ground_truth", "ground_truth_label"} & set(sample)
        record = deepcopy((findings, sample, scale_kind, allowed_scales))
        self.calls.append(("evaluate", record))
        self.evaluations.append(record)
        return self.decisions.pop(0)

    def conclude(self, evidence, trajectory, sample):
        assert not {"ground_truth", "ground_truth_label"} & set(sample)
        self.final_evidence = deepcopy(evidence)
        # This is the existing PaperModelBackend.conclude history contract.
        self.final_history = deepcopy([{k: e[k] for k in ("iteration", "action", "decision")}
                                       for e in trajectory])
        self.calls.append(("conclude", deepcopy((evidence, self.final_history, sample))))
        return {"answer": "A", "explanation": "reviewed evidence"}


def inputs(n=101, kind="physical", size=4096):
    scale = 5 if kind == "physical" else 1
    regions = [Region(f"{i:04}", "/unused", i * size, 0, size, size,
                      512, 512, kind, scale, .5) for i in range(n)]
    descriptions = {r.region_id: "generic" for r in regions}
    features = {r.region_id: [1., 0.] for r in regions}
    return regions, descriptions, features


def config():
    return {"algorithm": {"max_iterations": 5, "initial_ratio": .1,
                          "replenish_ratio": .05, "question_description_topk": 5,
                          "zoom_magnifications": [10, 20, 40], "relative_zoom_factors": [2, 4, 8]}}


def sample():
    return make_sample("test", "test", "one", "slide", "/unused", "Which?",
                       {"A": "one", "B": "two"}, "B")


def execute(decisions, enabled=True, n=101, kind="physical", size=4096):
    cfg = config()
    if enabled is not None:
        cfg["algorithm"]["zoom_return_to_loop"] = enabled
    backend = RecordingBackend(decisions)
    source = inputs(n, kind, size)
    original = deepcopy(source)
    result = run_agent(sample(), *source, backend, cfg)
    assert source == original
    return result, backend


@pytest.mark.parametrize("enabled", [None, False])
def test_disabled_wrapper_preserves_legacy_zoom_termination(enabled):
    decisions = [decision("zoom")]
    actual, backend = execute(decisions, enabled)
    legacy_backend = RecordingBackend(decisions)
    expected = _run_agent_legacy(sample(), *inputs(), legacy_backend, config())
    actual.pop("agent_seconds")
    expected.pop("agent_seconds")
    assert actual == expected
    assert backend.calls == legacy_backend.calls
    assert actual["stop_reason"] == "zoom_completed"
    assert "total_zoomed_regions" not in actual


@pytest.mark.parametrize("actions,n", [
    (["conclude"], 101),
    (["explore", "conclude"], 101),
    (["explore"] * 5, 101),
    (["explore"], 1),
])
def test_no_zoom_preserves_all_model_inputs_and_legacy_final_history(actions, n):
    decisions = [decision(action) for action in actions]
    legacy, old_backend = execute(decisions, False, n)
    looping, new_backend = execute(decisions, True, n)
    assert new_backend.calls == old_backend.calls
    assert new_backend.final_history == old_backend.final_history
    assert new_backend.final_evidence == old_backend.final_evidence
    for before, after in zip(legacy["process"], looping["process"]):
        assert {key: after[key] for key in before} == before
        assert not {"requested_action", "actual_action", "action_executed"} & set(after["decision"])
    for key in set(legacy) - {"process", "agent_seconds"}:
        assert looping[key] == legacy[key]
    if len(actions) == 5:
        assert looping["process"][-1]["action"] == "explore"
        assert looping["process"][-1]["actual_action"] is None
        assert looping["process"][-1]["action_executed"] is False


def test_first_zoom_keeps_all_parents_candidates_and_call_prefix_then_reviews():
    legacy, before = execute([decision("zoom", 40)], False)
    result, after = execute([decision("zoom", 40), decision("conclude")])
    assert after.calls[:len(before.calls) - 1] == before.calls[:-1]
    assert after.candidate_batches[0] == before.candidate_batches[0]
    assert len(after.candidate_batches[0]) == 11 * 64  # all 11 parents, not topk=5
    assert [e["action"] for e in result["process"]] == ["zoom", "conclude"]
    assert [len(e[0]) for e in after.evaluations] == [11, 12]
    assert len(after.described) == 6  # five low descriptions, one child
    assert after.final_history[-1]["decision"]["sufficient"] == "Yes"
    assert len(after.final_history) == 2
    assert result["stop_reason"] == "sufficient"
    assert result["total_examined_regions"] == legacy["total_examined_regions"] == 11
    assert result["total_zoomed_regions"] == 1


def test_repeated_zoom_uses_unvisited_siblings_and_never_redescribes_parents():
    result, backend = execute([decision("zoom"), decision("zoom"), decision("conclude")])
    first, second = [e["zoom_finding"] for e in result["process"][:2]]
    first_id, second_id = [f["region"]["region_id"] for f in (first, second)]
    assert first_id != second_id
    assert len(backend.candidate_batches[0]) == 44
    assert len(backend.candidate_batches[1]) == 43
    assert first_id not in {r["region_id"] for r in backend.candidate_batches[1]}
    assert {r["parent_id"] for r in backend.candidate_batches[1]} == {f"{i:04}" for i in range(11)}
    assert [len(e[0]) for e in backend.evaluations] == [11, 12, 13]
    assert len(backend.described) == 7
    assert result["process"][0]["active_zoom_regions"] == []
    assert result["process"][1]["active_zoom_regions"] == [first_id]
    assert result["process"][2]["active_zoom_regions"] == [first_id, second_id]
    ids = [f["region"]["region_id"] for f in backend.final_evidence]
    assert len(ids) == len(set(ids)) == 13


@pytest.mark.parametrize("kind,targets", [("physical", [40, 20]), ("relative", [8, 2])])
def test_changing_zoom_scale_retains_original_parents(kind, targets):
    result, backend = execute([decision("zoom", targets[0]), decision("zoom", targets[1]),
                               decision("conclude")], n=1, kind=kind)
    assert [e["zoom_finding"]["region"]["scale"] for e in result["process"][:2]] == targets
    for candidates in backend.candidate_batches:
        assert {r["parent_id"] for r in candidates} == {"0000"}
        assert all(r["region_id"].count("/z") == 1 for r in candidates)


def test_explore_resets_active_zoom_but_keeps_all_final_evidence():
    result, backend = execute([decision("zoom"), decision("explore"), decision("zoom"),
                               decision("conclude")])
    assert [e["action"] for e in result["process"]] == ["zoom", "explore", "zoom", "conclude"]
    assert [len(e[0]) for e in backend.evaluations] == [11, 12, 6, 7]
    old_child = result["process"][0]["zoom_finding"]["region"]["region_id"]
    assert old_child not in {f["region"]["region_id"] for f in backend.evaluations[2][0]}
    assert result["process"][2]["active_zoom_regions"] == []
    expected_parents = {f"{i:04}" for i in range(11, 17)}
    assert {r["parent_id"] for r in backend.candidate_batches[1]} == expected_parents
    final_ids = [f["region"]["region_id"] for f in backend.final_evidence]
    assert old_child in final_ids
    assert len(final_ids) == len(set(final_ids)) == 19
    assert result["total_examined_regions"] == 17
    assert result["total_zoomed_regions"] == 2


def test_five_round_budget_allows_four_zoom_acquisitions_and_reviews_last_child():
    result, backend = execute([decision("zoom")] * 5, n=1)
    assert len(backend.evaluations) == 5
    assert len(backend.candidate_batches) == 4
    assert [len(c) for c in backend.candidate_batches] == [4, 3, 2, 1]
    assert [len(e[0]) for e in backend.evaluations] == [1, 2, 3, 4, 5]
    assert len(backend.described) == 5
    assert result["total_zoomed_regions"] == 4
    assert result["stop_reason"] == "max_iterations"
    terminal = result["process"][-1]
    assert terminal["action"] == terminal["requested_action"] == "zoom"
    assert terminal["actual_action"] is None and terminal["action_executed"] is False
    assert "zoom_finding" not in terminal
    last_child = result["process"][-2]["zoom_finding"]
    assert last_child in backend.evaluations[-1][0]
    assert backend.final_history[-1]["decision"]["sufficient"] == "No"


def test_fifth_round_first_zoom_is_not_acquired_without_review_budget():
    decisions = [decision("explore")] * 4 + [decision("zoom")]
    result, backend = execute(decisions)
    assert len(backend.evaluations) == 5
    assert backend.candidate_batches == []
    assert len(backend.described) == 25
    assert result["total_zoomed_regions"] == 0
    assert result["stop_reason"] == "max_iterations"
    assert result["process"][-1]["action"] == "zoom"
    assert result["process"][-1]["action_executed"] is False


def test_no_valid_children_stops_without_fabricating_zoom_observation():
    result, backend = execute([decision("zoom")], n=1, size=1)
    assert result["stop_reason"] == "zoom_has_no_valid_children"
    assert backend.candidate_batches == []
    assert result["total_zoomed_regions"] == 0
    assert result["process"][0]["action_executed"] is False
    assert len(backend.final_evidence) == 1


def test_exhausted_children_stop_without_reusing_a_previous_observation(monkeypatch):
    monkeypatch.setattr("models.agent.zoom_regions", lambda parent, target: zoom_regions(parent, target)[:1])
    result, backend = execute([decision("zoom"), decision("zoom")], n=1)
    assert result["stop_reason"] == "zoom_has_no_unvisited_children"
    assert len(backend.candidate_batches) == 1
    assert len(backend.described) == 2
    terminal = result["process"][-1]
    assert terminal["zoom_candidate_count"] == 1
    assert terminal["zoom_unvisited_candidate_count"] == 0
    assert terminal["action_executed"] is False


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_wrapper_rejects_non_boolean_toggle(bad):
    cfg = config()
    cfg["algorithm"]["zoom_return_to_loop"] = bad
    backend = RecordingBackend([])
    with pytest.raises(ValueError, match="must be a boolean"):
        run_agent(sample(), *inputs(), backend, cfg)
    assert backend.calls == []


@pytest.mark.parametrize("enabled", [False, True])
def test_unsupported_zoom_and_missing_information_remain_errors(enabled):
    with pytest.raises(ValueError, match="Unsupported zoom"):
        execute([decision("zoom", 15)], enabled)
    invalid = decision("explore")
    invalid["missing_info"] = " "
    with pytest.raises(ValueError, match="Missing missing_info"):
        execute([invalid], enabled)
