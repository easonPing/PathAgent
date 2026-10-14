from copy import deepcopy
from types import SimpleNamespace

import pytest

from data_processing.datasets import make_sample
from data_processing.regions import Region
from models.agent import run_agent


class Backend:
    def __init__(self, decisions):
        self.decisions = deepcopy(decisions)
        self.tissue_status, self.evidence_artifacts = {}, {}
        self.described, self.evaluated, self.contexts = [], [], []
        self.final = None

    def encode_text(self, text):
        return [1., 0.]

    def encode_regions(self, regions):
        return [[1., 0.] for _ in regions]

    def source(self, region):
        return SimpleNamespace(width=100000, height=100000)

    def describe(self, region, question, missing_info=None):
        self.described.append((region.region_id, missing_info))
        return 'observed ' + region.region_id

    def evaluate(self, findings, sample, kind, allowed, review_context=None):
        assert 'ground_truth' not in sample
        self.evaluated.append(deepcopy(findings))
        self.contexts.append(deepcopy(review_context))
        return self.decisions.pop(0)

    def conclude(self, evidence, trajectory, sample, review_context=None):
        self.final = deepcopy(evidence)
        return {'answer': 'A', 'explanation': 'evidence reviewed'}


def decision(action, target=None, level=20, observed='Unknown'):
    return {'answer': 'A', 'thinking_steps': 'morphology',
            'sufficient': 'Yes' if action == 'conclude' else 'No',
            'missing_info': 'cell cohesion', 'zoom_recommendation': 'Yes' if action == 'zoom' else 'No',
            'zoom_level': level if action == 'zoom' else None, 'target_region': target,
            'target_observed': observed}


def execute(tmp_path, decisions, variant='R5_TARGET_REGION', n=11, size=4096):
    cfg = {'protocol': 'pathology-rules-v2', 'rules_v2': {'variant': variant},
           'runtime': {'processed_root': str(tmp_path)},
           'algorithm': {'max_iterations': 5, 'initial_ratio': .1, 'replenish_ratio': .05,
                         'question_description_topk': 5, 'zoom_selected_children': 5,
                         'zoom_magnifications': [10, 20, 40]}}
    regions = [Region('r%03d' % i, '/not-read', i * size, 0, size, size,
                      1024, 1024, 'physical', 5., .5, 'assumed') for i in range(n)]
    descriptions = {r.region_id: 'generic' for r in regions}
    features = {r.region_id: [1., 0.] for r in regions}
    row = make_sample('test', 'test', 'q', 'slide', '/not-read', 'Which?',
                      {'A': 'one', 'B': 'two'}, 'B')
    backend = Backend(decisions)
    before = deepcopy(descriptions)
    result = run_agent(row, regions, descriptions, features, backend, cfg)
    assert descriptions == before
    return result, backend


def test_top_five_are_reviewed_and_bound_to_requested_parent(tmp_path):
    result, backend = execute(tmp_path, [decision('zoom', 'r001'), decision('conclude')])
    event = result['process'][0]
    assert len(event['zoom_findings']) == 5
    assert event['executed_parent_regions'] == ['r001']
    assert all(f['region']['parent_id'] == 'r001' for f in event['zoom_findings'])
    assert len(backend.evaluated[1]) == 7
    assert backend.contexts[1]['last_request']['actual_action'] == 'zoom'
    assert result['total_zoomed_regions'] == 5


def test_fewer_than_five_children_all_reviewed(tmp_path):
    result, backend = execute(tmp_path, [decision('zoom', 'r000', 10), decision('conclude')], n=1)
    assert len(result['process'][0]['zoom_findings']) == 4
    assert len(backend.evaluated[1]) == 5


def test_cumulative_evidence_survives_explore_and_historical_target(tmp_path):
    result, backend = execute(tmp_path, [decision('zoom', 'r000'), decision('explore'),
                                       decision('zoom', 'r000'), decision('conclude')])
    assert len(backend.evaluated[1]) == 7
    assert len(backend.evaluated[2]) == 8
    assert 'r000' in {r['region_id'] for r in backend.contexts[2]['eligible_zoom_parents']}
    assert result['process'][2]['executed_parent_regions'] == ['r000']
    first = set(result['process'][0]['selected_children'])
    second = set(result['process'][2]['selected_children'])
    assert not first.intersection(second)


def test_review_budget_leaves_fifth_request_unexecuted(tmp_path):
    result, backend = execute(tmp_path, [decision('zoom', 'r000', 40)] * 5, n=1)
    assert len(result['process']) == 5
    assert result['total_zoomed_regions'] == 20
    assert result['process'][-1]['action_executed'] is False
    assert result['process'][-1]['actual_action'] is None
    assert result['stop_reason'] == 'max_iterations'


def test_invalid_parent_is_never_silently_retargeted(tmp_path):
    from models.inference import ModelProtocolError
    with pytest.raises(ModelProtocolError, match='not eligible'):
        execute(tmp_path, [decision('zoom', 'r010')])


def test_empty_target_pool_falls_back_to_explore_explicitly(tmp_path):
    result, backend = execute(tmp_path, [decision('zoom', 'r000'), decision('conclude')], size=1)
    event = result['process'][0]
    assert event['requested_target_region'] == 'r000'
    assert event['actual_action'] == 'explore'
    assert event['fallback_action'] == 'explore'
    assert not event.get('zoom_findings')
    assert len(backend.evaluated[1]) == 3


def test_untargeted_zoom_uses_current_parents(tmp_path):
    result, _ = execute(tmp_path, [decision('zoom', level=10), decision('conclude')])
    assert result['process'][0]['zoom_candidate_count'] == 8
    assert len(result['process'][0]['zoom_findings']) == 5


def test_no_progress_counts_executed_acquisitions_only(tmp_path):
    result, backend = execute(tmp_path, [decision('zoom'), decision('explore', observed='No'),
                                       decision('conclude', observed='No')], variant='R4_VISUAL_MISSING')
    assert [e['no_progress_count'] for e in result['process']] == [0, 1, 2]
    assert backend.contexts[2]['no_progress_count'] == 1


def test_quality_filter_does_not_fill_top_five_with_invalid_children(tmp_path, monkeypatch):
    def quality(source, region):
        rejected = region.parent_id is not None and not region.region_id.endswith('/0_0')
        return {'region_id': region.region_id, 'hard_reject': rejected,
                'reason': 'blank' if rejected else 'kept'}
    monkeypatch.setattr('models.rules_v2_agent.quality_for_region', quality)
    result, _ = execute(tmp_path, [decision('zoom', level=20), decision('conclude')],
                        variant='R3_ROI_VALIDITY')
    assert len(result['process'][0]['zoom_findings']) == 2
    assert result['quality_summary']['hard_rejected'] == 30


def test_all_rejected_regions_still_produce_answer_and_record_status(tmp_path, monkeypatch):
    monkeypatch.setattr('models.rules_v2_agent.quality_for_region',
                        lambda s, r: {'region_id': r.region_id, 'hard_reject': True, 'reason': 'outside'})
    result, backend = execute(tmp_path, [], variant='R3_ROI_VALIDITY')
    assert result['pred_answer'] == 'A'
    assert backend.final == []
    assert result['stop_reason'] == 'no_valid_low_regions'
    assert result['quality_summary']['hard_rejected'] == 11
