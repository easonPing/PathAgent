"""Pure, versioned prompt composition for the approved BCNB rule ablation.

Original paper/context templates remain immutable. This module changes only
the templates selected by a rule and synchronizes each stage's output shape.
"""
import copy
import hashlib
import json


PROTOCOL = 'pathology-rules-v2'
RULE_VERSION = 'rules-v2'
VARIANTS = ('R1_EVIDENCE', 'R2_DIFFERENTIAL', 'R3_ROI_VALIDITY',
            'R4_VISUAL_MISSING', 'R5_TARGET_REGION', 'ALL5')
RULES = ('R1', 'R2', 'R3', 'R4', 'R5')
REQUIRED_SCALES = {'overview': 5, 'architecture': 10,
                   'cellular': 20, 'fine_detail': 40}
_VARIANT_FLAGS = {name: frozenset((rule,))
                  for name, rule in zip(VARIANTS[:-1], RULES)}
_VARIANT_FLAGS['ALL5'] = frozenset(RULES)

SCALE_POLICY = (
    'Minimum evidence scales: overview (distribution, localization and '
    'representativeness) 5x; architecture (glands, nests and large structural '
    'relationships) 10x, or 20x if needed; cellular (cohesion, arrangement '
    'and nuclei) 20x, or 40x if needed; fine_detail (fine details and mitoses) '
    '40x. Higher magnification does not replace appropriate field coverage.'
)
BASE_FINAL_REQUIREMENT = (
    'The answer should directly reflect the information observable in the '
    'slide, such as biomarker expression level, presence or absence of '
    'features, or a numeric measurement.'
)
QUESTION_OUTPUT_CONTRACT = (
    'Return a compact JSON object inside <answer>...</answer>, with exactly '
    'the keys "tissue_status" and "findings". tissue_status is exactly '
    '"present", "absent" or "uncertain"; findings is a nonempty string '
    'describing visible findings. Escape quotation marks inside findings.'
)
QUESTION_FORMAT_RETRY = (
    'Correct the response format using the same image and observations. '
    + QUESTION_OUTPUT_CONTRACT
    + '\nRequired form:\n<answer>{{"tissue_status": "[one allowed value]", '
      '"findings": "[visible findings]"}}</answer>'
    + '\nReplace both bracketed placeholders with actual values. '
      'The previous response was:\n{previous_response}'
)
JSON_FORMAT_RETRY = (
    'Correct the previous response to match this stage contract. '
    'Required JSON fields and types: {schema}.\n'
    'Allowed choices and execution context: {context}.\n'
    '{stage_instruction}\nPrevious response:\n{previous_response}\n'
    'Return ONLY one compact valid JSON object.'
)


def flags_for_variant(name):
    if name not in _VARIANT_FLAGS:
        raise ValueError('Unknown rules-v2 variant: ' + str(name))
    return _VARIANT_FLAGS[name]


def _flags(flags):
    if isinstance(flags, str):
        return flags_for_variant(flags)
    flags = frozenset(flags)
    if not flags <= frozenset(RULES):
        raise ValueError('Unknown rules-v2 flags: ' + repr(sorted(flags)))
    return flags


def predict_keys(flags):
    flags = _flags(flags)
    keys = ['answer', 'thinking_steps']
    if 'R1' in flags:
        keys.append('required_features')
    if 'R2' in flags:
        keys.extend(('alternative_answer', 'discriminating_feature', 'conflicts'))
    return tuple(keys)


def reflect_keys(flags):
    flags = _flags(flags)
    keys = ['sufficient']
    if 'R1' in flags:
        keys.append('checks')
    if 'R2' in flags:
        keys.append('unresolved_conflicts')
    if 'R4' in flags:
        keys.append('target_observed')
    return tuple(keys)


def missing_keys(flags):
    flags = _flags(flags)
    keys = ['missing_info', 'zoom_recommendation',
            'recommended_zoom_level', 'zoom_reason']
    if 'R4' in flags:
        keys.append('evidence_type')
    if 'R5' in flags:
        keys.append('target_region')
    return tuple(keys)


def schemas_for_flags(flags):
    """JSON-serializable field contract used by messages and audit manifests."""
    flags = _flags(flags)
    predict = {'answer': 'string', 'thinking_steps': 'string'}
    reflect = {'sufficient': 'Yes|No'}
    missing = {'missing_info': 'string', 'zoom_recommendation': 'Yes|No',
               'recommended_zoom_level': '10|20|40 required for Yes; optional and unused for No', 'zoom_reason': 'string'}
    if 'R1' in flags:
        predict['required_features'] = [{'feature': 'string',
                                         'evidence_type': '|'.join(REQUIRED_SCALES)}]
        reflect['checks'] = [{'feature': 'string',
                              'evidence_type': '|'.join(REQUIRED_SCALES),
                              'region_id': 'string', 'assessable': 'boolean',
                              'coverage_sufficient': 'boolean'}]
    if 'R2' in flags:
        predict.update({'alternative_answer': 'string',
                        'discriminating_feature': 'string',
                        'conflicts': [{'region_id': 'string', 'feature': 'string',
                                       'claims': ['first opposing claim (string)',
                                                  'second opposing claim (string)']}]})
        reflect['unresolved_conflicts'] = [{'region_id': 'string', 'feature': 'string'}]
    if 'R4' in flags:
        missing['evidence_type'] = '|'.join(REQUIRED_SCALES)
        reflect['target_observed'] = 'Yes|No|Unknown'
    if 'R5' in flags:
        missing['target_region'] = 'optional eligible parent Region ID|null for Yes; unused for No'
    result = {'predict': predict, 'reflect': reflect, 'missing': missing,
              'final': {'answer': 'string', 'explanation': 'string'}}
    if 'R3' in flags:
        result['question'] = {'tissue_status': 'present|absent|uncertain',
                              'findings': 'string'}
    return result


def _baseline_templates():
    from models import paper_prompts, context_prompts
    result = {name: text for name, text in vars(paper_prompts).items()
              if name.endswith(('_SYSTEM', '_USER', '_PROMPT')) and isinstance(text, str)}
    for name in ('PERCEPTOR_USER', 'ZOOM_MISSING_USER', 'MISSING_USER'):
        result[name] = getattr(context_prompts, name)
    return result


def _without_output(system):
    marker = 'Output ONLY a JSON object:'
    if marker not in system:
        raise ValueError('Baseline system output instruction changed')
    return system.split(marker, 1)[0].rstrip()


def _output_contract(schema):
    return ('Output ONLY a JSON object with the following field contract '
            '(types and allowed values are shown; emit actual values): '
            + json.dumps(schema, ensure_ascii=False, separators=(',', ':')) + '.')


def templates_for_config(config):
    """Return effective templates without mutating original modules or config."""
    variant = config['rules_v2']['variant']
    flags = flags_for_variant(variant)
    templates = _baseline_templates()
    changed_schema = {'PREDICT_SYSTEM': bool(flags & {'R1', 'R2'}),
                      'REFLECT_SYSTEM': bool(flags & {'R1', 'R2', 'R4'}),
                      'MISSING_SYSTEM': bool(flags & {'R4', 'R5'})}
    for name, changed in changed_schema.items():
        if changed:
            templates[name] = _without_output(templates[name])

    def append(name, text):
        templates[name] += '\n\n' + text

    if 'R1' in flags:
        append('PREDICT_SYSTEM',
               'List one to four required_features groups with short feature names '
               '(at most eight words), covering the question\'s required evidence '
               'and every feature used in concise thinking_steps; assign an '
               'evidence_type from the supplied scale policy. Assign mitoses, '
               'mitotic figures, mitotic activity and mitotic counts to fine_detail.')
        templates['REFLECT_SYSTEM'] = (
            'You are an expert AI pathology assistant. For H&E biomarker or '
            'molecular labels, assess an indirect morphological estimate. '
            'For other questions, judge whether the current patch descriptions '
            'are sufficient to confidently support the answer.'
        )
        append('REFLECT_SYSTEM',
               'Emit exactly one check per required feature. '
               'For each required_features feature, copy its exact feature string '
               'into one check with one anchor region_id and emit compact JSON. '
               'Assess field/counting coverage using all available Regions and '
               'metadata as the scale authority; return No when required evidence '
               'is missing, unassessable, lacks required coverage or is below '
               'its minimum magnification.')
        append('MISSING_SYSTEM',
               'Inspect scale-insufficient features at or above their required '
               'magnification. Explore another low-power region when representation '
               'or field coverage is missing.')
        append('FINAL_SYSTEM',
               'State any unresolved evidence limitations in the explanation.')

    if 'R2' in flags:
        templates['PERCEPTOR_SYSTEM'] = (
            'You are a pathology image assistant. Describe visible H&E morphology '
            'at the supplied scale. Separate observations from interpretation and '
            'mark unassessable features. For biomarker or molecular questions, '
            'describe morphology relevant to an indirect estimate.'
        )
        templates['GENERIC_PROMPT'] = (
            'Describe the tissue, architecture and assessable cellular '
            'features in this H&E image.'
        )
        templates['QUESTION_PROMPT'] = (
            'Describe observable features relevant to the question: {question}. '
            'Separate assessed features from those that remain unassessable.'
        )
        append('PREDICT_SYSTEM',
               'For H&E biomarker or molecular labels, predict the best-fitting '
               'option as an indirect morphological estimate. Compare two '
               'distinct listed choices using observed features and identify their '
               'discriminating feature. Record both claims only for answer-relevant '
               'mutually exclusive observations about the same feature in the same pixel region; '
               'otherwise use an empty conflicts list.')
        append('REFLECT_SYSTEM',
               'Return No when an unresolved same-region, same-feature contradiction '
               'affects the answer. Omissions and differences between regions are '
               'not themselves contradictions. A lower-scale unassessable feature '
               'and a higher-scale observation can form an evidence update. '
               'Use an empty unresolved_conflicts list when none remain.')
        append('MISSING_SYSTEM',
               'Express a conflict as a neutral observable feature to recheck. '
               'Name the disputed Region in zoom_reason.')
        # Replace only the original final requirement, preserving all other
        # baseline text and the R1 suffix in ALL5.
        if templates['FINAL_SYSTEM'].count(BASE_FINAL_REQUIREMENT) != 1:
            raise ValueError('Baseline FINAL_SYSTEM requirement changed')
        templates['FINAL_SYSTEM'] = templates['FINAL_SYSTEM'].replace(
            BASE_FINAL_REQUIREMENT,
            'For H&E biomarker or molecular labels, return the best-fitting option '
            'as an indirect morphological estimate and state its basis and '
            'uncertainty. For other questions: ' + BASE_FINAL_REQUIREMENT)
        append('FINAL_SYSTEM',
               'Resolve conflicts using observations of the relevant feature. '
               'Explain any answer change through assessed evidence or a corrected '
               'inference; state material conflicts that remain unresolved.')
        append('SUMMARY_SYSTEM',
               'Keep observation sources separate and preserve unassessable '
               'features and material conflicts.')

    if 'R3' in flags:
        append('QUESTION_PROMPT',
               'Classify visible tissue as present, absent or uncertain. '
               'Describe question-relevant findings within visible tissue. '
               'Return JSON with tissue_status and findings inside <answer>...</answer>.')
        templates['QUESTION_OUTPUT_SYSTEM'] = QUESTION_OUTPUT_CONTRACT
        templates['QUESTION_OUTPUT_USER'] = QUESTION_OUTPUT_CONTRACT
        templates['QUESTION_FORMAT_RETRY_USER'] = QUESTION_FORMAT_RETRY
        append('REFLECT_SYSTEM',
               'Use assessable tissue as evidence. Background or padding does '
               'not establish feature absence or a negative label.')
        append('FINAL_SYSTEM',
               'Treat missing assessable tissue as unknown, not a negative label.')

    if 'R4' in flags:
        append('MISSING_SYSTEM',
               'Request an observable feature that distinguishes the plausible '
               'options. For H&E biomarker estimates, request morphology rather '
               'than assay results. Choose Explore for broader representation '
               'and Zoom for local detail. After two acquisitions without the '
               'requested evidence, choose a different feature, parent region '
               'or action.')
        append('REFLECT_SYSTEM',
               'Check whether the newly acquired descriptions provide the '
               'previously requested feature. Use target_observed=Unknown on '
               'the first round or when this cannot be determined.')
        append('FINAL_SYSTEM',
               'When retrieval adds no target evidence, use the available '
               'morphology and state the remaining uncertainty.')

    if 'R5' in flags:
        append('MISSING_SYSTEM',
               'For a location-specific check, set target_region to an eligible '
               'parent Region ID. Otherwise use null or omit it. '
               'For Zoom No, target_region is unused.')

    append('MISSING_SYSTEM',
           'For Zoom Yes, recommended_zoom_level must be an available integer. '
           'For Zoom No, recommended_zoom_level is optional and unused.')

    schemas = schemas_for_flags(flags)
    templates['JSON_FORMAT_RETRY_USER'] = JSON_FORMAT_RETRY
    templates['PREDICT_FORMAT_RETRY_USER'] = 'Use a listed answer choice when choices are provided.'
    templates['REFLECT_FORMAT_RETRY_USER'] = 'Use the required fields and actual JSON booleans.'
    templates['MISSING_FORMAT_RETRY_USER'] = 'Use an available Zoom level and the required fields.'
    templates['FINAL_FORMAT_RETRY_USER'] = (
        'Return one listed answer choice or label. For H&E biomarker or molecular '
        'questions, return the best-fitting indirect estimate and put uncertainty '
        'in explanation. Missing assay data is not a Negative result.'
    )
    if 'R1' in flags:
        templates['PREDICT_FORMAT_RETRY_USER'] += (
            ' Keep one to four short required_features groups covering the used evidence.'
        )
        templates['REFLECT_FORMAT_RETRY_USER'] += (
            ' Copy feature strings and evidence types exactly from required_features. '
            'Use one check per required feature with one region_id anchor; '
            'assess coverage using all available evidence.'
        )
    if 'R2' in flags:
        templates['PREDICT_FORMAT_RETRY_USER'] += (
            ' answer and alternative_answer must be two distinct listed choices. '
            'Each conflict must contain both opposing claims about the same feature '
            'in the same region; otherwise use conflicts=[].'
        )
    if 'R5' in flags:
        templates['MISSING_FORMAT_RETRY_USER'] += (
            ' For Zoom Yes, target_region must be an eligible parent ID, null or omitted. '
            'For Zoom No, target_region and recommended_zoom_level are unused.'
        )
    for name, stage in (('PREDICT_SYSTEM', 'predict'),
                        ('REFLECT_SYSTEM', 'reflect'),
                        ('MISSING_SYSTEM', 'missing')):
        if changed_schema[name]:
            append(name, _output_contract(schemas[stage]))
    if changed_schema['PREDICT_SYSTEM']:
        templates['PREDICT_USER'] = templates['PREDICT_USER'].replace(
            "Now output the JSON with 'answer' and 'thinking_steps'.",
            'Now output the JSON with all required fields.')
    if changed_schema['REFLECT_SYSTEM']:
        templates['REFLECT_USER'] = templates['REFLECT_USER'].replace(
            'Only return a JSON object: {{"sufficient": "Yes" or "No"}}',
            'Only return the JSON object with all required fields.')
    return templates


def prompt_manifest(config):
    """Full provenance, hashes, rule selection and stage schemas for freezing."""
    from models.context_prompts import prompt_manifest as original_manifest
    original = copy.deepcopy(original_manifest())
    baseline = _baseline_templates()
    effective = templates_for_config(config)
    variant = config['rules_v2']['variant']
    flags = flags_for_variant(variant)
    result = {**original, 'protocol': PROTOCOL, 'rule_version': RULE_VERSION,
              'variant': variant, 'rules': [r for r in RULES if r in flags],
              'post_zoom_reflect': True, 'zoom_selected_children': 5,
              'required_scales': dict(REQUIRED_SCALES),
              'scale_policy': SCALE_POLICY,
              'schemas': schemas_for_flags(flags), 'templates': {},
              'original_template_hashes': {}, 'effective_template_hashes': {}}
    for name, text in effective.items():
        if name not in baseline:
            actual_hash = hashlib.sha256(text.encode()).hexdigest()
            result['templates'][name] = {'text': text, 'sha256': actual_hash,
                                        'published': False, 'source': 'local rules-v2 stage format contract',
                                        'base_hash': None, 'base_source': None, 'base_published': False}
            result['original_template_hashes'][name] = None
            result['effective_template_hashes'][name] = actual_hash
            continue
        base_item = original['templates'][name]
        base_hash = hashlib.sha256(baseline[name].encode()).hexdigest()
        actual_hash = hashlib.sha256(text.encode()).hexdigest()
        item = copy.deepcopy(base_item)
        item.update({'text': text, 'sha256': actual_hash})
        if text != baseline[name]:
            item.update({'published': False,
                         'source': 'local rules-v2 template',
                         'base_hash': base_hash,
                         'base_source': base_item['source'],
                         'base_published': base_item['published']})
        result['templates'][name] = item
        result['original_template_hashes'][name] = base_hash
        result['effective_template_hashes'][name] = actual_hash
    result['notes'] = list(original.get('notes', [])) + [
        'All new variants use batch Zoom top-5 and post-Zoom Predict/Reflect.',
        'Original paper and context template modules are unchanged.',
        'Code-supplied Region metadata is authoritative; BCNB scale is assumed.',
        'On the unchanged single Predict format retry only, an omitted thinking_steps reporting field may be an empty string; original response and normalization are logged. Required rule fields and the final listed-choice answer remain strict.',
    ]
    if flags.intersection({'R1', 'R2'}):
        result['notes'].append('After exactly two failed Reflect format attempts, sufficiency is unknown and the program requests further evidence as No. Both original responses and protocol-format-unknown provenance are retained; no truncated prefix is treated as a model judgment.')
    if 'R2' in flags:
        result['notes'].append('A structurally complete Predict refusal or unknown conflict Region is preserved as a semantic evidence failure requiring review; no clinical answer or Region is substituted. Final must still select an actual listed choice.')
    if 'R3' in flags:
        result['question_wire_format'] = 'JSON inside answer; explicit tissue_status/findings tags accepted for compatibility'
        result['notes'].append('Explicit valid tissue states and narrow findings quotation repairs retain their original response. After the unchanged single format retry, completed nonempty prose without a valid state is retained as uncertain with source protocol-format-unknown; present/absent are never inferred. Incomplete JSON, empty or truncated output still fails.')
    return result
