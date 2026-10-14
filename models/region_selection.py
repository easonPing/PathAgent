"""Explicit region-count policies; the archived paper policy is the default."""
import math


POLICIES = ('paper', 'max5')


def selection_policy(algorithm):
    policy = algorithm.get('region_selection_policy', 'paper')
    if policy not in POLICIES:
        raise ValueError(f'Unknown region selection policy: {policy}')
    return policy


def low_region_count(algorithm, total, *, initial):
    ratio = algorithm['initial_ratio' if initial else 'replenish_ratio']
    count = math.ceil(total * ratio)
    return max(5, count) if selection_policy(algorithm) == 'max5' else count


def question_region_count(algorithm, available):
    if selection_policy(algorithm) == 'max5':
        return max(5, available)
    return algorithm['question_description_topk']


def zoom_region_count(algorithm, available):
    if selection_policy(algorithm) == 'max5':
        return max(5, available)
    return algorithm.get('zoom_selected_children', 1)


def selection_manifest(algorithm):
    return {
        'policy': selection_policy(algorithm),
        'initial_requested': 'max(5, ceil(initial_ratio * N))',
        'explore_requested': 'max(5, ceil(replenish_ratio * N))',
        'question_descriptions_requested': 'max(5, selected_low_region_count)',
        'zoom_children_requested': 'max(5, valid_unvisited_child_count)',
        'available_count_cap': True,
        'N': 'total initial low-magnification candidate regions before quality filtering',
        'effective_question_descriptions': 'all selected low-magnification regions',
        'effective_zoom_children': 'all valid unvisited Zoom children in the eligible parents',
    }
