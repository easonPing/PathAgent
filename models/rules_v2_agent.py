"""Opt-in cumulative review, top-five acquisitions, and auditable ROI binding."""
import hashlib
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from data_processing.common import atomic_json, digest, read_json, resolve
from data_processing.datasets import public_input
from data_processing.regions import zoom_regions
from data_processing.roi_quality import QUALITY_VERSION, quality_for_region
from models.agent import retrieve, unit_rows
from models.inference import ModelProtocolError
from models.rules_v2_prompts import flags_for_variant
from models.region_selection import low_region_count, question_region_count, zoom_region_count


def _scores(names, embeddings, query, selected):
    values = (unit_rows(embeddings) @ unit_rows(query).T).reshape(-1)
    by_id = dict(zip(names, values))
    return [{'region_id': name, 'rank': i + 1, 'score': float(by_id[name])}
            for i, name in enumerate(selected)]


def run_agent_v2(sample, regions, descriptions, features, backend, config):
    sample = public_input(sample)
    flags = flags_for_variant(config['rules_v2']['variant'])
    algorithm = config['algorithm']
    names = [r.region_id for r in regions]
    if not names or len(names) != len(set(names)):
        raise ValueError('Need nonempty unique frozen Regions')
    if set(names) != set(descriptions) or set(names) != set(features):
        raise ValueError('Frozen observation coverage differs')
    if len({r.scale_kind for r in regions}) != 1 or len({r.scale for r in regions}) != 1:
        raise ValueError('Initial Regions must use one scale')
    by_id = {r.region_id: r for r in regions}
    generic = deepcopy(descriptions)
    quality, tissue = {}, {}
    ledger, trajectory = {}, []
    visited, visited_zoom = set(), set()
    kind = regions[0].scale_kind
    allowed = algorithm['zoom_magnifications'] if kind == 'physical' else algorithm['relative_zoom_factors']
    total = len(names)
    current, stop_reason = [], None
    fresh_low_batch = True
    last_request, last_new, no_progress = None, [], 0
    started = time.monotonic()

    def quality_of(region):
        if region.region_id in quality:
            return quality[region.region_id]
        identity = {'version': QUALITY_VERSION, 'region': region.to_dict(),
                    'inputs': config['runtime'].get('rules_v2_initial_inputs_hash')}
        path = resolve(config['runtime']['processed_root']) / 'quality' / (digest(identity) + '.json')
        if path.exists():
            item = read_json(path)
            if item['identity'] != identity:
                raise ValueError('ROI quality cache changed')
            result = item['quality']
        else:
            result = quality_for_region(backend.source(region), region)
            atomic_json(path, {'identity': identity, 'quality': result})
        quality[region.region_id] = result
        return result

    def valid(pool):
        if 'R3' not in flags:
            return list(pool)
        return [region for region in pool if not quality_of(region)['hard_reject']]

    def retrieve_low(pool, query, count):
        candidates = valid([by_id[name] for name in pool])
        eligible = [r.region_id for r in candidates]
        if not eligible:
            return []
        return retrieve(eligible, [features[name] for name in eligible], backend.encode_text(query), count)

    def finding(region, generic_text='', question_text=''):
        status = getattr(backend, 'tissue_status', {}).get(region.region_id, 'uncertain')
        tissue[region.region_id] = status
        description = generic_text + ('\n' if generic_text and question_text else '') + question_text
        item = {'region': region.to_dict(), 'description': description,
                'generic_description': generic_text, 'question_description': question_text,
                'tissue_status': status}
        if 'R3' in flags:
            item['tissue_status_source'] = getattr(backend, 'tissue_status_source', {}).get(
                region.region_id, 'not_assessed')
        artifact = getattr(backend, 'evidence_artifacts', {}).get(region.region_id)
        if (not artifact and not question_text and hasattr(backend, 'save_evidence_image')
                and config['runtime'].get('rules_v2_evidence_image_root')):
            image = backend.source(region).read(region)
            try:
                artifact = {'image_artifact_path': backend.save_evidence_image(image),
                            'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                            'input_size': list(image.size), 'input_mode': image.mode,
                            'input_origin': 'cached_generic_crop'}
            finally:
                image.close()
        if artifact:
            item['image_artifact'] = deepcopy(artifact)
        ledger[region.region_id] = item
        return item

    def context():
        parents = [by_id[name].to_dict() for name in names if name in visited
                   and not quality.get(name, {}).get('hard_reject', False)]
        return {'eligible_zoom_parents': parents, 'last_request': deepcopy(last_request),
                'current_zoom_parents': [by_id[name].to_dict() for name in current],
                'last_new_findings': deepcopy(last_new), 'no_progress_count': no_progress,
                'quality': deepcopy(quality), 'tissue_status': deepcopy(tissue)}

    current = retrieve_low(names, sample['question'], low_region_count(algorithm, total, initial=True))
    for iteration in range(1, algorithm['max_iterations'] + 1):
        if not current:
            stop_reason = 'no_valid_low_regions' if 'R3' in flags else 'no_unexamined_regions'
            break
        if fresh_low_batch:
            if visited.intersection(current):
                raise RuntimeError('Already observed low Region selected')
            visited.update(current)
            new = []
            for index, name in enumerate(current):
                question_text = backend.describe(by_id[name], sample['question']) \
                    if index < question_region_count(algorithm, len(current)) else ''
                new.append(finding(by_id[name], generic[name], question_text))
            last_new = new
            fresh_low_batch = False
        findings = deepcopy(list(ledger.values()))
        decision = backend.evaluate(findings, sample, kind, allowed, review_context=context())
        if 'R4' in flags and last_request is not None:
            observed = str(decision.get('target_observed', 'Unknown')).lower()
            no_progress = 0 if observed == 'yes' else no_progress + 1
        event = {'iteration': iteration, 'findings': findings, 'decision': decision,
                 'examined_region_count': len(visited), 'current_parent_regions': list(current),
                 'active_zoom_regions': [name for name in ledger if name in visited_zoom],
                 'no_progress_count': no_progress}
        trajectory.append(event)
        if str(decision.get('sufficient', '')).lower() == 'yes':
            event.update(action='conclude', requested_action='conclude', actual_action='conclude',
                         action_executed=True)
            stop_reason = 'sufficient'
            break
        missing = str(decision.get('missing_info') or '').strip()
        if not missing:
            raise ModelProtocolError('Missing an executable information request')
        zoom = str(decision.get('zoom_recommendation', '')).lower() == 'yes'
        action = 'zoom' if zoom else 'explore'
        event.update(action=action, requested_action=action, actual_action=None, action_executed=False)
        if iteration == algorithm['max_iterations']:
            stop_reason = 'max_iterations'
            event['failure_reason'] = 'no_remaining_review_round'
            break
        request = {'missing_info': missing, 'evidence_type': decision.get('evidence_type'),
                   'requested_action': action, 'target_region': decision.get('target_region')}
        if zoom:
            target = float(decision['zoom_level'])
            if target not in allowed:
                raise ModelProtocolError('Unsupported Zoom level')
            requested_parent = decision.get('target_region') if 'R5' in flags else None
            eligible_ids = {r['region_id'] for r in context()['eligible_zoom_parents']}
            if requested_parent is not None and requested_parent not in eligible_ids:
                raise ModelProtocolError('Requested Zoom parent is not eligible')
            parents = [requested_parent] if requested_parent is not None else list(current)
            all_children = [child for name in parents for child in zoom_regions(by_id[name], target)]
            unvisited = [child for child in all_children if child.region_id not in visited_zoom]
            candidates = valid(unvisited)
            event.update(requested_target_region=requested_parent,
                         zoom_candidate_count=len(all_children),
                         zoom_unvisited_candidate_count=len(unvisited),
                         zoom_valid_candidate_count=len(candidates), zoom_query=missing)
            if candidates:
                candidate_ids = [r.region_id for r in candidates]
                embeddings, query = backend.encode_regions(candidates), backend.encode_text(missing)
                selected = retrieve(candidate_ids, embeddings, query, zoom_region_count(algorithm, len(candidates)))
                child_by_id = {r.region_id: r for r in candidates}
                acquired = []
                for name in selected:
                    child = child_by_id[name]
                    if requested_parent is not None and child.parent_id != requested_parent:
                        raise RuntimeError('Executed child violates target_region')
                    question_text = backend.describe(child, sample['question'], missing_info=missing)
                    acquired.append(finding(child, question_text=question_text))
                    visited_zoom.add(name)
                event.update(zoom_findings=deepcopy(acquired), selected_children=selected,
                             executed_parent_regions=sorted({child_by_id[name].parent_id for name in selected}),
                             retrieval_scores=_scores(candidate_ids, embeddings, query, selected),
                             actual_action='zoom', action_executed=True)
                request['actual_action'] = 'zoom'
                last_request, last_new = request, acquired
                continue
            event['failure_reason'] = 'target_has_no_available_children'
            # The unsuccessful location request remains recorded. Explore is explicit.
            event['fallback_action'] = 'explore'
        remaining = [name for name in names if name not in visited]
        next_regions = retrieve_low(remaining, missing, low_region_count(algorithm, total, initial=False))
        event['next_regions'] = next_regions
        if not next_regions:
            stop_reason = 'no_valid_unexamined_regions' if 'R3' in flags else 'no_unexamined_regions'
            if zoom:
                event['failure_reason'] = 'target_has_no_children_and_no_explore_regions'
            break
        event.update(actual_action='explore', action_executed=True)
        request['actual_action'] = 'explore'
        last_request, current, fresh_low_batch = request, next_regions, True
    evidence = deepcopy(list(ledger.values()))
    final = backend.conclude(evidence, trajectory, sample, review_context=context())
    rejects = Counter(q.get('reason', 'unknown') for q in quality.values() if q.get('hard_reject'))
    return {'pred_answer': final['answer'], 'explanation': final['explanation'], 'status': 'ok',
            'process': trajectory, 'stop_reason': stop_reason, 'final_response': final,
            'initial_region_count': total, 'total_examined_regions': len(visited),
            'total_zoomed_regions': len(visited_zoom), 'total_evidence_regions': len(evidence),
            'quality_summary': {'checked': len(quality), 'hard_rejected': sum(rejects.values()),
                                'rejection_reasons': dict(rejects)},
            'roi_quality': quality, 'generic_provenance': deepcopy(getattr(backend, 'generic_provenance', {})),
            'agent_seconds': time.monotonic() - started}
