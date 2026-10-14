"""Verify acquired images and model inputs, including archived-count compatibility."""
from copy import deepcopy
import sys

import pytest
import yaml

from data_processing.common import load_config
from data_processing.datasets import make_sample
from data_processing.regions import Region
from models.agent import run_agent
from models.protocols import protocol_manifest


class Backend:
    def __init__(self, decisions):
        self.decisions = deepcopy(decisions)
        self.described, self.evaluated = [], []
        self.tissue_status, self.evidence_artifacts = {}, {}

    def encode_text(self, text):
        return [1., 0.]

    def encode_regions(self, regions):
        return [[1., 0.] for _ in regions]

    def source(self, region):
        return None  # Quality tests replace the reader with an explicit validity oracle.

    def describe(self, region, question, missing_info=None):
        self.described.append((region.to_dict(), missing_info))
        return 'observed ' + region.region_id

    def evaluate(self, findings, sample, kind, allowed, review_context=None):
        assert 'ground_truth' not in sample
        self.evaluated.append(deepcopy(findings))
        return self.decisions.pop(0)

    def conclude(self, evidence, trajectory, sample, review_context=None):
        self.final = deepcopy(evidence)
        return {'answer': 'A', 'explanation': 'reviewed'}


def decision(action, level=20, target=None):
    return {'answer': 'A', 'thinking_steps': 'morphology',
            'sufficient': 'Yes' if action == 'conclude' else 'No',
            'missing_info': 'nuclear detail',
            'zoom_recommendation': 'Yes' if action == 'zoom' else 'No',
            'zoom_level': level, 'target_region': target}


def execute(tmp_path, n, decisions, flow='v2', variant='R5_TARGET_REGION', policy='max5'):
    cfg = {'protocol': 'pathology-rules-v2' if flow == 'v2' else 'paper-prompts-context-restored-v3',
           'rules_v2': {'variant': variant}, 'runtime': {'processed_root': str(tmp_path)},
           'algorithm': {'region_selection_policy': policy, 'max_iterations': 5,
                         'initial_ratio': .1, 'replenish_ratio': .1 if policy == 'max5' else .05,
                         'question_description_topk': 5, 'zoom_selected_children': 5 if flow == 'v2' else 1,
                         'zoom_return_to_loop': flow == 'loop', 'zoom_magnifications': [10, 20, 40]}}
    regions = [Region(f'r{i:03d}', '/unused', i * 4096, 0, 4096, 4096,
                      1024, 1024, 'physical', 5., .5, 'assumed') for i in range(n)]
    descriptions = {r.region_id: 'generic' for r in regions}
    features = {r.region_id: [1., 0.] for r in regions}
    sample = make_sample('test', 'test', 'q', 'slide', '/unused', 'Which?',
                         {'A': 'one', 'B': 'two'}, 'B')
    original = deepcopy(descriptions)
    backend = Backend(decisions)
    result = run_agent(sample, regions, descriptions, features, backend, cfg)
    assert descriptions == original
    return result, backend


@pytest.mark.parametrize('flow', ['v2', 'legacy', 'loop'])
@pytest.mark.parametrize('n,expected', [(3, 3), (11, 5), (101, 11)])
def test_initial_describes_every_selected_region(tmp_path, flow, n, expected):
    result, backend = execute(tmp_path, n, [decision('conclude')], flow)
    assert len(backend.evaluated[0]) == len(backend.described) == expected
    assert result['total_examined_regions'] == expected


@pytest.mark.parametrize('flow', ['v2', 'legacy', 'loop'])
@pytest.mark.parametrize('n,first,second', [(7, 5, 2), (11, 5, 5), (101, 11, 11)])
def test_explore_uses_ten_percent_and_describes_all_new_regions(tmp_path, flow, n, first, second):
    result, backend = execute(tmp_path, n, [decision('explore'), decision('conclude')], flow)
    assert len(backend.evaluated[0]) == first
    assert len(result['process'][0]['next_regions']) == second
    assert result['total_examined_regions'] == first + second
    assert len(backend.described) == first + second
    described = [r['region_id'] for r, _ in backend.described]
    assert len(described) == len(set(described))
    assert {f['region']['region_id'] for f in backend.final} == set(described)


@pytest.mark.parametrize('flow', ['v2', 'legacy', 'loop'])
@pytest.mark.parametrize('n,level,expected', [(1, 10, 4), (11, 20, 16)])
def test_zoom_acquires_all_available_children_and_keeps_them_in_evidence(tmp_path, flow, n, level, expected):
    decisions = [decision('zoom', level, 'r000')]
    if flow != 'legacy':
        decisions.append(decision('conclude'))
    result, backend = execute(tmp_path, n, decisions, flow)
    if flow != 'v2' and n == 11:
        expected *= 5  # Legacy-style navigation searches all five current parents.
    acquired = result['process'][0]['zoom_findings']
    assert len(acquired) == expected
    assert len({f['region']['region_id'] for f in acquired}) == expected
    described_children = [r for r, missing in backend.described if missing is not None]
    assert len(described_children) == expected
    assert all(f in backend.final for f in acquired)
    if flow != 'legacy':
        assert all(f in backend.evaluated[1] for f in acquired)


def test_exhausted_zoom_parent_falls_back_to_fresh_explore_without_duplicates(tmp_path):
    result, backend = execute(tmp_path, 11, [decision('zoom', target='r000'),
                                           decision('zoom', target='r000'), decision('conclude')])
    assert len(result['process'][0]['zoom_findings']) == 16
    assert result['process'][1]['zoom_unvisited_candidate_count'] == 0
    assert result['process'][1]['actual_action'] == 'explore'
    assert len(result['process'][1]['next_regions']) == 5
    ids = [r['region_id'] for r, _ in backend.described]
    assert len(ids) == len(set(ids)) == 26


def test_quality_rejection_is_not_bypassed_to_reach_five(tmp_path, monkeypatch):
    def quality(source, region):
        rejected = region.parent_id is not None and not region.region_id.endswith(('/0_0', '/0_1'))
        return {'region_id': region.region_id, 'hard_reject': rejected,
                'reason': 'blank' if rejected else 'kept'}
    monkeypatch.setattr('models.rules_v2_agent.quality_for_region', quality)
    result, _ = execute(tmp_path, 1, [decision('zoom'), decision('conclude')], variant='R3_ROI_VALIDITY')
    assert result['process'][0]['zoom_valid_candidate_count'] == 2
    assert len(result['process'][0]['zoom_findings']) == 2
    assert result['quality_summary']['hard_rejected'] == 14


def test_low_quality_filter_uses_all_three_valid_regions_without_padding(tmp_path, monkeypatch):
    monkeypatch.setattr('models.rules_v2_agent.quality_for_region',
                        lambda source, region: {'region_id': region.region_id,
                                                'hard_reject': region.region_id not in {'r000', 'r001', 'r002'},
                                                'reason': 'outside'})
    result, backend = execute(tmp_path, 11, [decision('conclude')], variant='R3_ROI_VALIDITY')
    assert len(backend.described) == result['total_examined_regions'] == 3


def test_new_config_is_distinct_and_prompt_text_is_unchanged():
    old = load_config('configs/reproduce_v3.yaml')
    new = load_config('configs/reproduce_v3_max5.yaml')
    assert old['algorithm']['replenish_ratio'] == .05
    assert new['algorithm']['replenish_ratio'] == .1
    assert new['algorithm']['region_selection_policy'] == 'max5'
    assert new['slurm']['shards'] == 10 and new['slurm']['max_concurrent_gpus'] is None
    original, updated = protocol_manifest(old), protocol_manifest(new)
    assert original['templates'] == updated['templates']
    assert 'region_selection' not in original
    assert updated['region_selection']['effective_zoom_children'].startswith('all valid unvisited')


@pytest.mark.parametrize('change', [{'region_selection_policy': 'unknown'}, {'replenish_ratio': .05}])
def test_invalid_policy_or_old_explore_ratio_is_rejected(tmp_path, change):
    cfg = load_config('configs/reproduce_v3_max5.yaml')
    cfg['algorithm'].update(change)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize('arguments,expected', [([], 'max5'), (['--region-selection-policy', 'paper'], 'paper')])
def test_new_campaign_cli_defaults_to_max5_and_can_reproduce_paper_counts(tmp_path, monkeypatch, arguments, expected):
    from slurm import submit_rule_ablations_v2 as launch
    calls = []
    monkeypatch.setattr(launch, 'prepare', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['submit_rule_ablations_v2.py', 'prepare', '--root', str(tmp_path), *arguments])
    launch.main()
    assert calls == [{'region_selection_policy': expected}]
