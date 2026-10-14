"""Concise local HTML and complete metrics for the six BCNB rules-v2 arms."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import html
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import report_rule_ablations as old

FAMILY = ('R1_EVIDENCE', 'R2_DIFFERENTIAL', 'R3_ROI_VALIDITY',
          'R4_VISUAL_MISSING', 'R5_TARGET_REGION', 'ALL5')
DATASET = 'bcnb_1000_sample'
LABELS = {'BASELINE': 'Baseline', 'R1_EVIDENCE': 'R1 证据',
          'R2_DIFFERENTIAL': 'R2 鉴别', 'R3_ROI_VALIDITY': 'R3 有效区域',
          'R4_VISUAL_MISSING': 'R4 目标', 'R5_TARGET_REGION': 'R5 位置', 'ALL5': 'ALL'}


def paired_bootstrap(samples, baseline, results, iterations=10000, seed=128, family_size=6):
    if iterations <= 0:
        raise ValueError('Bootstrap iterations must be positive')
    result = {'iterations': iterations, 'seed': seed, 'unit': 'image_id', 'family_size': family_size,
        'family_alpha': .05, 'method': 'paired image-cluster percentile bootstrap',
        'family_confidence_percent': 100 * (1 - .05 / family_size),
        'limitation': 'Bonferroni intervals for six prespecified overall comparisons; approximate bootstrap coverage, no rerun variance. Subset intervals are exploratory.'}
    if any(row['sample_id'] not in baseline or row['sample_id'] not in results for row in samples):
        return {**result, 'status': 'withheld_incomplete'}
    import numpy as np
    clusters = defaultdict(lambda: [0, 0, 0])
    for row in samples:
        sid = row['sample_id']
        cluster = clusters[str(row['image_id'])]
        cluster[0] += 1
        cluster[1] += int(old.outcome(row, baseline[sid])['correct'])
        cluster[2] += int(old.outcome(row, results[sid])['correct'])
    data = np.asarray([clusters[key] for key in sorted(clusters)], dtype=float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, iterations, 256):
        indices = rng.integers(0, len(data), size=(min(256, iterations - start), len(data)))
        totals = data[indices].sum(axis=1)
        draws.append(100 * (totals[:, 2] - totals[:, 1]) / totals[:, 0])
    values = np.concatenate(draws)
    alpha = .05 / family_size
    return {**result, 'status': 'complete', 'clusters': len(data), 'questions': len(samples),
        'delta_accuracy_percent': float(100 * (data[:, 2].sum() - data[:, 1].sum()) / data[:, 0].sum()),
        'delta_ci95_percent': np.quantile(values, [.025, .975]).tolist(),
        'delta_family_ci_percent': np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist()}


def zoom_findings(event):
    if 'zoom_findings' in event:
        return event.get('zoom_findings') or []
    return [event['zoom_finding']] if event.get('zoom_finding') else []


def rule_metrics(samples, results):
    counts, checks, types, targets, quality, stages, scales = (Counter() for _ in range(7))
    affected = defaultdict(list)
    generic_by_image = {}
    usage = old.empty_usage()
    for row in samples:
        sid = row['sample_id']
        result = results.get(sid)
        if result is None:
            continue
        events = result.get('process') or []
        provenance = result.get('generic_provenance') or {}
        if provenance:
            record = generic_by_image.setdefault(str(row['image_id']), {'calls': 0, 'seconds': 0.})
            record['calls'] = max(record['calls'], provenance.get('generated_calls_this_process', 0))
            record['seconds'] = max(record['seconds'], provenance.get('generated_seconds_this_process', 0.))
        flags = set()
        counts['received_questions'] += 1
        counts['review_rounds'] += len(events)
        if events:
            yes = str(events[0].get('decision', {}).get('sufficient', '')).lower() == 'yes'
            counts['first_reflect_yes'] += yes
            counts['first_reflect_yes_final_wrong'] += yes and not old.outcome(row, result)['correct']
        seen_regions = set()
        for event_index, event in enumerate(events):
            decision = event.get('decision') or {}
            action = event.get('actual_action', event.get('action'))
            executed = event.get('action_executed', action in ('zoom', 'explore'))
            counts['reflect_' + str(decision.get('sufficient', 'unknown')).lower()] += 1
            if executed and action in ('zoom', 'explore'):
                counts['actual_' + action + '_batches'] += 1
                flags.add('actual_' + action)
            for check in decision.get('checks') or []:
                if isinstance(check, dict):
                    checks['entries'] += 1
                    checks['assessable_yes' if check.get('assessable') is True else 'assessable_no'] += 1
                    checks['coverage_yes' if check.get('coverage_sufficient') is True else 'coverage_no'] += 1
                    types[str(check.get('evidence_type', 'unknown'))] += 1
            failures = decision.get('scale_failures') or []
            for failure in decision.get('evidence_failures') or []:
                reason = failure.get('reason', failure.get('type', 'unspecified')) if isinstance(failure, dict) else str(failure)
                checks['failure_' + str(reason)] += 1
                if reason == 'invalid_conflict_source':
                    counts['invalid_conflict_sources'] += 1
                    flags.add('invalid_conflict_source')
                if reason == 'invalid_primary_answer':
                    counts['primary_refusal_events'] += 1
                    flags.add('primary_refusal')
            counts['scale_failure_events'] += bool(failures)
            counts['scale_failures'] += len(failures)
            if failures:
                flags.add('scale_failure')
            raw = decision.get('reflection_raw')
            if decision.get('reflection_source') == 'protocol-format-unknown':
                counts['reflect_format_fallback_events'] += 1
                flags.add('reflect_format_fallback')
            if isinstance(raw, dict) and 'sufficient' in raw:
                counts['raw_reflect_' + str(raw['sufficient']).lower()] += 1
                counts['raw_yes_effective_no'] += (str(raw.get('sufficient')).lower() == 'yes'
                                                  and str(decision.get('sufficient')).lower() == 'no')
            conflicts = decision.get('conflicts') or []
            unresolved = decision.get('unresolved_conflicts') or []
            counts['conflict_events'] += bool(conflicts)
            counts['unresolved_conflict_events'] += bool(unresolved)
            if unresolved:
                flags.add('unresolved_conflict')
            if 'target_observed' in decision:
                previous = events[event_index - 1] if event_index else None
                previous_action = (previous or {}).get('actual_action', (previous or {}).get('action'))
                previous_executed = (previous or {}).get('action_executed',
                                    previous_action in ('zoom', 'explore'))
                if previous_executed and previous_action in ('zoom', 'explore'):
                    targets['observed_' + str(decision['target_observed']).lower()] += 1
                    targets['post_acquisition_assessments'] += 1
                else:
                    targets['target_observed_without_acquisition'] += 1
            if decision.get('missing_info'):
                counts['missing_requests'] += 1
                types['missing_' + str(decision.get('evidence_type', 'unknown'))] += 1
            if decision.get('target_region'):
                targets['generated_specified_requests'] += 1
                flags.add('target_generated')
            children = zoom_findings(event)
            counts['zoom_images'] += len(children)
            target = event.get('requested_target_region')
            if target:
                targets['specified_requests'] += 1
                flags.add('target_specified')
                if children:
                    match = all(child['region'].get('parent_id') == target for child in children)
                    targets['executed_requests'] += 1
                    targets['matching_requests'] += match
                    targets['mismatching_requests'] += not match
                    if not match:
                        flags.add('target_mismatch')
            if children:
                counts['zoom_batches_with_saved_images'] += 1
                if event is not events[-1]:
                    next_event = events[event_index + 1]
                    next_ids = {f.get('region', {}).get('region_id') for f in next_event.get('findings', [])}
                    counts['post_zoom_review_contains_children'] += all(
                        f['region']['region_id'] in next_ids for f in children)
                for finding in children:
                    scales[old.scale_key(finding['region'].get('scale'))] += 1
            for finding in (event.get('findings') or []) + children:
                region = finding.get('region') or {}
                rid = region.get('region_id')
                if not rid or rid in seen_regions:
                    continue
                seen_regions.add(rid)
                tissue = finding.get('tissue_status')
                source = finding.get('tissue_status_source')
                if source:
                    quality['tissue_status_source_' + source] += 1
                if tissue:
                    quality['described_tissue_' + str(tissue)] += 1
                    if tissue == 'absent':
                        flags.add('tissue_absent')
        summary = result.get('quality_summary') or {}
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                quality[key] += value
            elif key == 'rejection_reasons' and isinstance(value, dict):
                for reason, count in value.items():
                    quality['rejected_' + reason] += count
        for call in result.get('model_calls') or []:
            stage = str(call.get('stage', 'unknown'))
            stages[stage] += 1
            old.add_call(usage, call)
        counts['evidence_regions'] += len(seen_regions)
        counts['target_no_progress_events'] += sum(bool(event.get('no_progress_count',
                                                      event.get('decision', {}).get('current_no_progress_count', 0)) >= 2)
                                                  for event in events)
        if result.get('audit_failures'):
            flags.add('audit_failure')
        for flag in flags:
            affected[flag].append(sid)
    return {'counts': dict(counts), 'check_status': dict(checks), 'evidence_types': dict(types),
            'targets': dict(targets), 'quality': dict(quality), 'model_call_stages': dict(stages),
            'qa_usage': usage, 'zoom_scales': dict(scales),
            'generic_generation': {'fresh_calls': sum(value['calls'] for value in generic_by_image.values()),
                                   'fresh_seconds': sum(value['seconds'] for value in generic_by_image.values()),
                                   'by_image': generic_by_image,
                                   'note': 'Deduplicated per arm/image from runtime provenance. Shared cache costs may fall on another arm/process.'},
            'affected_questions': {k: sorted(v) for k, v in affected.items()},
            'definitions': {'quality': 'Runtime image checks and model tissue_status; not independent pixel truth.',
                            'target_observed': 'Model assessment after an actually executed preceding Zoom/Explore; initial or unexecuted requests excluded, subject to manual review.',
                            'count_overlap': 'Question flags and failure modes can overlap.'}}


def build_report(root, iterations=10000, seed=128):
    global _OLD_IMAGES, _OLD_IMAGE_PATH
    root = Path(root).resolve()
    plan = old.read_json(root / 'plan.json')
    from slurm.submit_rule_ablations_v2 import audit_full_results, FULL_COMPLETED
    from slurm import submit_rule_ablations as controller_legacy
    state_path = root / 'status.json'
    controller_states = old.read_json(state_path).get('variants', {}) if state_path.exists() else None
    frozen_error = None
    try:
        controller_legacy.verify_frozen({'files': plan.get('files', {})})
    except (ValueError, FileNotFoundError) as exc:
        frozen_error = str(exc)
    _OLD_IMAGES = None
    _OLD_IMAGE_PATH = Path(plan['baseline_root']).resolve().parents[2] / 'runs/analysis/bcnb_failure_modes_20261009/roi_images.json'
    if tuple(plan['variants']) != FAMILY:
        raise ValueError('Report expects exactly six approved rules-v2 arms')
    manifest = old.load_manifest(plan['datasets'][DATASET]['full_manifest'])
    samples = manifest['samples']
    arms = {'BASELINE': old.load_arm(plan['baseline_root'], DATASET, samples)}
    for name, spec in plan['variants'].items():
        expected = {**spec, **spec['datasets'][DATASET]}
        arms[name] = old.load_arm(spec['root'], DATASET, samples, expected)
    baseline = arms['BASELINE']['results']
    if old.group_metrics(samples, baseline)['correct'] != plan['datasets'][DATASET].get('baseline_expected_correct', 513):
        raise ValueError('Archived baseline correct count differs')
    groups = old.sample_groups(samples)
    report = {'schema_version': 2, 'generated_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'root': str(root), 'plan_hash': old.digest(plan), 'samples_hash': manifest['samples_hash'],
        'complete': True, 'baseline_reused': True, 'arms': {}, 'manual_review': {'status': 'pending'},
        'comparison_limitation': plan.get('comparison_limitation'),
        'validation_readiness_policy': plan.get('validation_readiness_policy'),
        'source_version_note': plan.get('source_version_note'),
        'source_versions': {name: spec.get('source_version', {
            'campaign_root': str(root), 'validation_source_hash': spec['source_hash'],
            'full_source_hash': spec['source_hash'], 'validation_root': str(Path(spec['root']) / DATASET / 'validation'),
            'full_root': str(Path(spec['root']) / DATASET / 'full')}) for name, spec in plan['variants'].items()},
        'bootstrap': {'iterations': iterations, 'seed': seed, 'family_size': 6,
                      'unit': 'image_id', 'family_confidence_percent': 100 * (1 - .05 / 6)}}
    rows, cases, case_ids = [], [], set()
    contracts, completion_states = {}, {}
    for name, arm in arms.items():
        results = arm['results']
        metrics = {group: old.group_metrics(group_rows, results) for group, group_rows in groups.items()}
        report['complete'] &= metrics['overall']['complete']
        if name == 'BASELINE':
            contract = {'passed': True, 'source': 'verified_archived_baseline'}
            completion = 'complete'
        else:
            status = (controller_states or {}).get(name, {})
            try:
                if frozen_error:
                    raise ValueError(frozen_error)
                contract = audit_full_results(plan, name, results)
                if status.get('stage') == 'full_failed':
                    raise ValueError(status.get('error', 'Controller full execution failed'))
                completion = contract['completion_status']
                if controller_states is not None and status.get('stage') not in FULL_COMPLETED:
                    contract.update(passed=None, reason='Controller full verification is pending')
                    completion = 'verification_pending'
            except (ValueError, FileNotFoundError, KeyError) as exc:
                contract = {'passed': False, 'error': str(exc)}
                completion = 'full_failed'
            contract['controller_stage'] = status.get('stage', 'standalone_contract_audit')
        contracts[name], completion_states[name] = contract, completion
        paired = {}
        for group, group_rows in groups.items():
            transitions = Counter({key: 0 for key in old.TRANSITIONS})
            for row in group_rows:
                sid = row['sample_id']
                if sid in results and sid in baseline:
                    transitions[old.transition(old.outcome(row, baseline[sid])['correct'],
                                               old.outcome(row, results[sid])['correct'])] += 1
            ci = None
            if name != 'BASELINE':
                ci = paired_bootstrap(group_rows, baseline, results, iterations, seed,
                    family_size=6 if group == 'overall' else 1) if contract['passed'] is True else {
                        'status': 'withheld_execution_contract', 'error': contract.get('error', contract.get('reason'))}
            if ci is not None:
                ci['exploratory'] = group != 'overall'
            paired[group] = {'transitions': dict(transitions), 'bootstrap': ci}
            rows.append({'variant': name, 'group': group, **metrics[group],
                         'gained': transitions['wrong_correct'], 'lost': transitions['correct_wrong'],
                         'delta': (ci or {}).get('delta_accuracy_percent'),
                         'family_ci': (ci or {}).get('delta_family_ci_percent'),
                         'completion_status': completion})
        rules = rule_metrics(samples, results)
        trajectory = old.trajectory_metrics(samples, arm)
        # The legacy counter knows only one Zoom child; v2 rule_metrics above is authoritative for batches/images.
        trajectory['note'] = 'Legacy cost/prediction counters retained; rules_metrics is authoritative for v2 Zoom batches and images.'
        report['arms'][name] = {'metrics': metrics, 'paired_with_baseline': paired,
            'rules_metrics': rules, 'cost_and_trajectory': trajectory, 'provenance': arm['provenance'],
            'execution_contract': contract, 'completion_status': completion}
        anomalies = set(rules['affected_questions'].get('audit_failure', []))
        anomalies.update(rules['affected_questions'].get('target_mismatch', []))
        anomalies.update(rules['affected_questions'].get('invalid_conflict_source', []))
        anomalies.update(rules['affected_questions'].get('reflect_format_fallback', []))
        anomalies.update(rules['affected_questions'].get('primary_refusal', []))
        for sample in samples:
            sid = sample['sample_id']
            left, right = old.outcome(sample, baseline.get(sid)), old.outcome(sample, results.get(sid))
            change = (left['label'] != right['label']) if right['state'] != 'missing' else False
            category = old.transition(left['correct'], right['correct']) if right['state'] != 'missing' else 'missing'
            record = {'sample_id': sid, 'variant': name, 'task': sample.get('task'),
                'image_id': sample['image_id'], 'truth': sample['choices'][sample['ground_truth_label']],
                'baseline': sample['choices'].get(left['label'], left['state']),
                'prediction': sample['choices'].get(right['label'], right['state']),
                'transition': category, 'label_changed': change,
                'result_path': arm['paths'].get(sid), 'baseline_result_path': arms['BASELINE']['paths'].get(sid),
                'audit_anomaly': sid in anomalies, 'state': right['state']}
            if name != 'BASELINE':
                cases.append(record)
                if change or sid in anomalies or right['state'] == 'invalid':
                    case_ids.add(sid)
    data_complete = report['complete']
    execution_passed = all(value['passed'] is True for value in contracts.values())
    failed = any(value['passed'] is False for value in contracts.values())
    model_errors = any(arm['metrics']['overall']['invalid'] for arm in report['arms'].values())
    status = ('full_failed' if failed else 'incomplete_not_final' if not data_complete
              else 'data_complete_pending_verification' if not execution_passed
              else 'completed-with-model-errors' if model_errors else 'complete')
    report.update(status=status, complete=data_complete and execution_passed,
                  data_complete=data_complete, execution_contract_passed=execution_passed,
                  denominator_policy='Full manifest denominator; invalid and missing predictions count incorrect. Model-format errors are distinct from execution-contract failures.',
                  case_pages=len(case_ids), changed_records=sum(row['label_changed'] for row in cases))
    report['manual_review']['required_sample_ids'] = sorted(case_ids)
    report['manual_review']['required_changed_or_anomalous_records'] = [
        {'sample_id': row['sample_id'], 'variant': row['variant'], 'transition': row['transition']}
        for row in cases if row['label_changed'] or row['audit_anomaly'] or row['state'] == 'invalid']
    review_path = root / 'manual_review.json'
    if review_path.exists():
        reviewed = {(record['sample_id'], record['variant']) for record in old.read_json(review_path).get('records', [])
                    if record.get('reviewed') and record.get('analysis')}
        required = {(record['sample_id'], record['variant'])
                    for record in report['manual_review']['required_changed_or_anomalous_records']}
        report['manual_review'].update(status='complete' if required and required.issubset(reviewed) else 'pending',
                                      reviewed_records=len(required & reviewed))
    return report, rows, cases, case_ids, samples, arms


CSS = '''body{font:15px/1.55 system-ui,sans-serif;max-width:1280px;margin:28px auto;padding:0 20px;color:#182536;background:#f7f9fc}h1{font-size:26px}h2{font-size:19px;margin-top:26px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px 10px;border-bottom:1px solid #dde4ee;text-align:left}th{background:#eaf0f7;position:sticky;top:0}a{color:#175b9c}p.note{color:#526475;font-size:13px}select,input,button{padding:7px;margin:4px}details{background:white;border:1px solid #dde4ee;border-radius:6px;margin:8px 0;padding:10px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,monospace;max-height:540px;overflow:auto}img{width:240px;height:240px;object-fit:contain;background:white;border:1px solid #dde4ee}.images{display:flex;gap:12px;overflow:auto}.image-card{min-width:240px;font-size:12px}.gain{color:#16703b}.loss{color:#ad3232}.panel{overflow:auto;max-height:650px}.badge{font-size:12px;color:#526475}''' 


def esc(value):
    return html.escape(str(value), quote=True)


def case_filename(sid):
    return 'q' + sid.split(':')[-1] + '.html'


_OLD_IMAGES = None
_OLD_IMAGE_PATH = None
_VERIFIED_IMAGES = set()


def verified_native_image(call):
    path = call.get('image_artifact_path')
    if not path or not Path(path).is_file():
        return None
    stat = Path(path).stat()
    key = (path, stat.st_size, stat.st_mtime_ns, call.get('pixel_sha256'))
    if key not in _VERIFIED_IMAGES:
        from PIL import Image
        with Image.open(path) as image:
            if image.mode != 'RGB' or hashlib.sha256(image.tobytes()).hexdigest() != call.get('pixel_sha256'):
                raise ValueError('Model input artifact RGB does not match recorded pixels: ' + path)
            if list(image.size) != list(call.get('input_size', image.size)):
                raise ValueError('Model input artifact dimensions differ: ' + path)
        _VERIFIED_IMAGES.add(key)
    return path


def verified_thumbnail(call):
    global _OLD_IMAGES
    if _OLD_IMAGES is None:
        path = _OLD_IMAGE_PATH or (Path(__file__).resolve().parents[1] / 'runs/analysis/bcnb_failure_modes_20261009/roi_images.json')
        _OLD_IMAGES = {value['pixel_sha256']: value for value in old.read_json(path).values()
                       if value.get('pixel_verified')} if path.is_file() else {}
    info = _OLD_IMAGES.get(call.get('pixel_sha256'))
    if not info:
        return None
    path = Path(info['thumbnail'])
    if not path.is_file():
        return None
    from data_processing.common import sha256_file
    if sha256_file(path) != info['thumbnail_sha256']:
        raise ValueError('Verified baseline thumbnail changed: ' + str(path))
    return str(path)


def image_cards(result):
    cards, seen = [], set()
    calls = [call for call in result.get('model_calls') or [] if call.get('stage') == 'describe']
    for event in result.get('process') or []:
        for finding in (event.get('findings') or []) + zoom_findings(event):
            artifact = finding.get('image_artifact')
            if artifact:
                calls.append({**artifact, 'region': finding.get('region'),
                              'input_origin': artifact.get('input_origin',
                                  'question_description' if finding.get('question_description') else 'cached_generic_crop')})
    for call in calls:
        region = call.get('region') or {}
        rid = region.get('region_id')
        if rid in seen:
            continue
        seen.add(rid)
        path = verified_native_image(call)
        thumbnail = False
        if not path or not Path(path).is_file():
            path = verified_thumbnail(call)
            thumbnail = bool(path)
        img = '<img loading="lazy" src="' + esc(path) + '" alt="' + esc(rid) + '">' if path and Path(path).is_file() else ''
        source = region.get('image_path')
        source_link = '<a href="' + esc(source) + '">原始切片</a>' if source else ''
        origin = '缓存 Generic 对应区域 RGB；未新增描述调用。' if call.get('input_origin') == 'cached_generic_crop' else 'Question/Zoom 模型输入 RGB'
        cards.append('<div class="image-card">' + img + '<div>' + esc(rid) + ' · ' + esc(region.get('scale'))
            + '×</div><div>像素范围 ' + esc([region.get(k) for k in ('x', 'y', 'width', 'height')])
            + '</div>' + source_link + ('<div>已验证来源的缩略图；原输入像素 SHA 保存在记录中。</div>' if thumbnail else '<div>' + origin + '</div>' if img else '<div>无保存的模型输入图；使用原图链接及坐标。</div>') + '</div>')
    return '<div class="images">' + ''.join(cards) + '</div>' if cards else ''


def evidence_rounds(result):
    body = '<details><summary>各轮证据：Generic / Question</summary>'
    for index, event in enumerate(result.get('process') or []):
        body += '<details><summary>第 ' + esc(event.get('iteration', index + 1)) + ' 轮 · ' + esc(event.get('actual_action', event.get('action'))) + '</summary>'
        for finding in event.get('findings') or []:
            region = finding.get('region') or {}
            body += '<details><summary>' + esc(region.get('region_id')) + ' · ' + esc(region.get('scale')) + '× · ' + esc(finding.get('tissue_status', '未记录组织状态')) + '</summary>'
            if 'generic_description' in finding or 'question_description' in finding:
                body += '<p>Generic</p><pre>' + esc(finding.get('generic_description') or '未生成') + '</pre>'
                body += '<p>Question</p><pre>' + esc(finding.get('question_description') or '未生成') + '</pre>'
            else:
                body += '<p>原版本合并描述</p><pre>' + esc(finding.get('description', '')) + '</pre>'
            body += '</details>'
        body += '</details>'
    return body + '</details>'


def write_case(root, sample, arms):
    sid = sample['sample_id']
    body = '<h1>BCNB ' + esc(sid.split(':')[-1]) + '</h1><p><a href="../report.zh.html">返回总表</a></p>'
    body += '<p>' + esc(sample.get('question')) + '</p><p>' + esc(sample['choices']) + '</p><p>真值：' + esc(sample['choices'][sample['ground_truth_label']]) + '</p>'
    for name, arm in arms.items():
        result = arm['results'].get(sid)
        if not result:
            continue
        value = old.outcome(sample, result)
        body += '<details><summary>' + esc(LABELS[name]) + ' · ' + esc(result.get('pred_answer')) + ' · ' + ('正确' if value['correct'] else '错误') + '</summary>'
        if result.get('status') != 'ok':
            body += '<p>状态：' + esc(result.get('status')) + ' · ' + esc(result.get('error_stage')) + ' · ' + esc(result.get('error')) + '</p>'
            if not result.get('process'):
                body += '<p class="note">部分执行轨迹未序列化，未取得合法Final；以下保留实际可用模型原文与取证图，不将缺失轨迹解释为成功阅片。</p>'
        body += '<p>' + esc(result.get('explanation', '')) + '</p>' + image_cards(result)
        body += evidence_rounds(result)
        body += '<details><summary>' + ('完整执行轨迹' if result.get('process') else '未序列化的部分轨迹') + '</summary><pre>' + esc(json.dumps(result.get('process'), ensure_ascii=False, indent=2)) + '</pre></details>'
        for index, call in enumerate(result.get('model_calls') or []):
            title = str(index + 1) + '. ' + str(call.get('stage'))
            if call.get('region'):
                title += ' · ' + str(call['region'].get('region_id'))
            body += '<details><summary>' + esc(title) + '</summary><pre>' + esc(call.get('raw_response', call.get('answer_content', ''))) + '</pre></details>'
        body += '<p><a href="' + esc(arm['paths'][sid]) + '">原始结果及完整输入消息</a></p></details>'
    old.atomic_text(root / 'cases' / case_filename(sid), '<!doctype html><html lang="zh"><meta charset="utf-8"><title>BCNB ' + esc(sid) + '</title><style>' + CSS + '</style><body>' + body + '</body></html>')


def write_report(root, report, rows, cases, case_ids, samples, arms):
    root = Path(root)
    old.atomic_text(root / 'report.json', old.json_text(report))
    old.atomic_text(root / 'case_index.json', old.json_text(cases))
    review_path = root / 'manual_review.json'
    review = old.read_json(review_path) if review_path.exists() else {
        'status': 'pending', 'scope': 'All changed and anomalous records; images, every output and full trajectory.',
        'records': []}
    previous = {(record['sample_id'], record['variant']): record for record in review['records']}
    for record in report['manual_review']['required_changed_or_anomalous_records']:
        previous.setdefault((record['sample_id'], record['variant']),
                            {**record, 'reviewed': False, 'analysis': ''})
    review['records'] = list(previous.values())
    review['status'] = 'complete' if review['records'] and all(record.get('reviewed') and record.get('analysis')
                                                            for record in review['records']) else 'pending'
    old.atomic_text(review_path, old.json_text(review))
    for sample in samples:
        if sample['sample_id'] in case_ids:
            write_case(root, sample, arms)
    overall = [row for row in rows if row['group'] == 'overall']
    body = '<h1>BCNB1000 · rules-v2</h1><p>' + esc(report['status']) + ' · ' + esc(report['generated_at']) + '</p>'
    selection = old.read_json(root / 'plan.json').get('region_selection_policy', 'paper')
    flow = ('首轮和 Explore 至少5个或10%的区域、全部选中区域的 Question 描述、全部可用 Zoom 子图和放大后复核'
            if selection == 'max5' else 'Zoom top-5 和放大后复核')
    body += '<p class="note">复用原 baseline；六组共同采用' + flow + '。相对 baseline 的差异包含共同流程变化。程序指标与人工分析分开保存。</p>'
    if report.get('validation_readiness_policy'):
        body += '<p class="note">有限验证按固定28题收齐结果与执行合同判断就绪；已知模型格式失败保留并单列。人工审查覆盖真实可用图片、原文和已保存轨迹，缺失部分轨迹及无Final另行确认。</p>'
    body += '<p class="note">正确率固定使用完整 manifest 分母；invalid 和 missing 均计错并单列。completed-with-model-errors 表示取齐结果但存在模型格式错误；full_failed 表示执行合同失败，其指标仅用于诊断，不作为通过或提升结论。</p>'
    body += '<h2>总体</h2><table><thead><tr><th>组</th><th>完成</th><th>执行状态</th><th>Invalid / Missing</th><th>正确率</th><th>变化 / pp</th><th>改对 / 改错</th><th>六比较校正区间</th></tr></thead><tbody>'
    for row in overall:
        ci = row['family_ci']
        score_ready = row['complete'] and report['arms'][row['variant']]['execution_contract']['passed'] is True
        accuracy = format(row['accuracy_percent'], '.2f') + '%' if score_ready else '—'
        delta = format(row['delta'], '+.2f') if score_ready and row['delta'] is not None else '—'
        interval = format(ci[0], '.2f') + '～' + format(ci[1], '.2f') if score_ready and ci else '—'
        body += '<tr><td>' + esc(LABELS[row['variant']]) + '</td><td>' + str(row['received']) + '/' + str(row['expected']) + '</td><td>' + esc(row['completion_status']) + '</td><td>' + str(row['invalid']) + ' / ' + str(row['missing']) + '</td><td>' + accuracy + '</td><td>' + delta + '</td><td>' + str(row['gained']) + ' / ' + str(row['lost']) + '</td><td>' + interval + '</td></tr>'
    body += '</tbody></table><p class="note">区间：' + str(report['bootstrap']['iterations']) + ' 次按切片配对 bootstrap，Bonferroni 校正六项总体比较；未估计重复运行方差。</p>'
    tasks = sorted({row['group'][5:] for row in rows if row['group'].startswith('task:')})
    by_key = {(row['variant'], row['group']): row for row in rows}
    body += '<h2>Subset</h2><div class="panel"><table><thead><tr><th>Subset</th><th>N</th>' + ''.join('<th>' + esc(LABELS[name]) + '</th>' for name in ('BASELINE',) + FAMILY) + '</tr></thead><tbody>'
    for task in tasks:
        body += '<tr><td>' + esc(task) + '</td><td>' + str(by_key['BASELINE', 'task:' + task]['expected']) + '</td>'
        for name in ('BASELINE',) + FAMILY:
            row = by_key[name, 'task:' + task]
            score_ready = row['complete'] and report['arms'][name]['execution_contract']['passed'] is True
            body += '<td>' + (format(row['accuracy_percent'], '.2f') + '%' if score_ready else '—') + (' · invalid ' + str(row['invalid']) if row['invalid'] else '') + '</td>'
        body += '</tr>'
    body += '</tbody></table></div><h2>修改与完整指标</h2><p><a href="code_diff.patch">代码差异</a> · <a href="report.json">完整指标 JSON</a> · <a href="manual_review.json">人工审核记录</a></p><p>'
    body += ' · '.join('<a href="variants/' + esc(name) + '/prompt_diff.patch">' + esc(LABELS[name]) + ' prompt</a>' for name in FAMILY) + '</p>'
    if report.get('source_version_note'):
        body += '<details><summary>各组冻结源码与真实结果来源</summary><p class="note">R3/R4/R5 的 validation 和 full 使用同一旧冻结版本；R1/R2/ALL5 使用新冻结版本。跨组 source hash 不同，旧 validation 不核验新代码。</p><table><tr><th>组</th><th>冻结 campaign</th><th>Validation / full source hash</th><th>结果</th></tr>'
        for name, version in report['source_versions'].items():
            body += '<tr><td>' + esc(LABELS[name]) + '</td><td>' + esc(Path(version['campaign_root']).name) + '</td><td><code>' + esc(version['validation_source_hash'][:16]) + ' / ' + esc(version['full_source_hash'][:16]) + '</code></td><td><a href="variants/' + name + '/' + DATASET + '/validation/">validation</a> · <a href="variants/' + name + '/' + DATASET + '/full/">full</a></td></tr>'
        body += '</table><p><a href="source_versions.json">完整来源记录</a></p></details>'
    body += '<h2>改题与轨迹</h2><p><select id="arm"><option value="">全部组</option>' + ''.join('<option value="' + name + '">' + esc(LABELS[name]) + '</option>' for name in FAMILY) + '</select><select id="task"><option value="">全部 subset</option>' + ''.join('<option>' + esc(task) + '</option>' for task in tasks) + '</select><select id="kind"><option value="changed">标签变化或异常</option><option value="wrong_correct">改对</option><option value="correct_wrong">改错</option><option value="">全部结果</option></select><input id="search" placeholder="题号 / 类别"><span id="shown" class="badge"></span></p><div class="panel"><table><thead><tr><th>题号</th><th>组 / subset</th><th>真值</th><th>Baseline</th><th>新预测</th><th>变化</th><th>轨迹</th></tr></thead><tbody id="cases"></tbody></table></div>'
    body += '<p class="note">有标签变化或执行异常的题提供七组并排入口，展开查看图片、每步输出和轨迹；其他结果直接打开完整原始 JSON。人工审查完成前不将程序计数写成已确认的病理原因。</p>'
    data = json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')
    pages = json.dumps(sorted(case_ids), ensure_ascii=False).replace('<', '\\u003c')
    body += '''<script>const data=DATA;const pages=new Set(PAGES);const labels=LABELS;const $=id=>document.getElementById(id);const safe=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function render(){const arm=$('arm').value,task=$('task').value,kind=$('kind').value,search=$('search').value.toLowerCase();const rows=data.filter(r=>(!arm||r.variant===arm)&&(!task||r.task===task)&&(!kind||(kind==='changed'?(r.label_changed||r.audit_anomaly||r.state==='invalid'):r.transition===kind))&&(!search||JSON.stringify(r).toLowerCase().includes(search)));$('shown').textContent=rows.length+' 条';$('cases').innerHTML=rows.map(r=>{const cls=r.transition==='wrong_correct'?'gain':r.transition==='correct_wrong'?'loss':'';const href=pages.has(r.sample_id)?'cases/q'+r.sample_id.split(':').pop()+'.html':r.result_path;return '<tr class="'+cls+'"><td>'+safe(r.sample_id)+'</td><td>'+safe(labels[r.variant])+' / '+safe(r.task)+'</td><td>'+safe(r.truth)+'</td><td>'+safe(r.baseline)+'</td><td>'+safe(r.prediction)+'</td><td>'+safe(r.transition)+'</td><td>'+(href?'<a href="'+safe(href)+'">查看</a>':'待完成')+'</td></tr>';}).join('');}for(const id of ['arm','task','kind','search'])$(id).addEventListener('input',render);render();</script>'''.replace('DATA', data).replace('PAGES', pages).replace('LABELS', json.dumps(LABELS, ensure_ascii=False))
    old.atomic_text(root / 'report.zh.html', '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCNB1000 rules-v2</title><style>' + CSS + '</style><body>' + body + '</body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=128)
    args = parser.parse_args()
    values = build_report(args.root, args.bootstrap, args.seed)
    write_report(args.root, *values)
    print('Report:', Path(args.root).resolve() / 'report.zh.html')


if __name__ == '__main__':
    main()
