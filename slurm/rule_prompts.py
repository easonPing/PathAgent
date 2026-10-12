"""Compile isolated rules-v0 prompt variants without importing any model code.

The generated snapshots change prompt string constants and prompt-manifest
attribution only. They deliberately retain the existing inference protocol,
output keys, generation settings, and agent control flow.
"""
import ast
import hashlib
from pathlib import Path


PACKAGES = ('O', 'D', 'R', 'M', 'F')
VARIANTS = tuple(name + '_ONLY' for name in PACKAGES) + ('ALL',)
RULE_VERSION = 'rules-v0'
_METADATA_MARKER = '# RULES_V0_MANIFEST_METADATA'


def selected_packages(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown rules-v0 variant: ' + str(variant))
    return PACKAGES if variant == 'ALL' else (variant.removesuffix('_ONLY'),)


# These short, conditional instructions are experimental rules, not a claim
# that a particular objective magnification establishes diagnostic validity.
OBSERVATION = (
    'Report visible observations separately from interpretations. Start with '
    'architecture in this region, then relevant cellular features. Limit claims '
    'to the supplied region and distinguish present, not seen, and not assessable. '
    'The question directs attention; it does not supply a stain, assay result, '
    'organ identity, patient history, percentage, or calibrated count. Do not '
    'invent these. Preserve uncertainty and scale limitations. Give a concise '
    'observation report, not speculative scratch work.'
)
OBSERVATION_PREDICT = (
    'Separate observed morphology from interpretation and unprovided data. '
    'Treat speculative <think> text as hypotheses, not established observations. '
    'Generic and question-focused descriptions under one Region are repeated '
    'views of the same region, not independent patches or votes. Identify '
    'same-region contradictions; different regions may be heterogeneous. '
    'Unknown or unassessable is not negative, score 0, score 2+, or triple '
    'negative. Do not invent tests or measurements to justify a selected option.'
)
DIFFERENTIAL = (
    'Apply only the capsule matching the question and options. Compare the '
    'closest alternatives using their discriminating morphology; general '
    'malignancy alone does not establish a subtype. Preserve the benchmark '
    'option meanings and do not invent missing findings. '
    '[Lung] Distinguish true glands/acini from solid growth and seek actual '
    'squamous differentiation for a squamous alternative. Glands plus solid '
    'areas do not alone prove adenosquamous carcinoma. A local pattern does '
    'not establish whole-slide predominance. '
    '[Endometrium] Integrate architecture and nuclear features. Papillae, '
    'clear cytoplasm, or focal mucin alone do not prove serous, clear-cell, '
    'or mucinous subtype; consider endometrioid differentiation and mixed '
    'features. '
    '[Breast Tumor] Compare ductal/NST with lobular evidence such as '
    'discohesion and single-file/targetoid growth. General atypia or absence '
    'of a special structure does not by itself exclude ILC. Molecular '
    'subtypes are not histological alternatives to IDC/ILC/Other. '
    '[Breast Nottingham] Assess tubule formation, nuclear pleomorphism, '
    'and mitotic activity separately. Do not fill all three with highest '
    'scores from atypia alone, invent counts/HPF, or substitute another '
    'organ grading system. Missing components constrain the grade estimate. '
    '[Melanoma] Distinguish evidence for melanoma from evidence for its '
    'specific subtype. Generic and subtype options may overlap in hierarchy; '
    'do not invent subtype-defining features or silently merge labels. '
    '[Receptors/Molecular] Real ER/PR IHC is assessed in tumor nuclei: '
    'at least 1% positive nuclei can support positivity; weak staining is '
    'not automatically negative. ER low-positive reporting is not a PR '
    'category. For real HER2 IHC, intensity, membrane completeness and '
    'proportion all matter: strong complete staining in >10% supports 3+; '
    'weak/moderate complete staining in >10% can be 2+, not an unknown '
    'placeholder. IHC score and amplification/final HER2 status are distinct. '
    'Do not infer ER/PR/HER2/Ki67 results from H&E as observed assays. '
    'Do not equate an intrinsic molecular subtype with an IHC surrogate or '
    'invent receptor combinations. If a forced choice exceeds the evidence, '
    'label it an uncertain indirect estimate in the existing explanation '
    'or thinking_steps field.'
)
REFLECTION = (
    'Audit the current evidence independently of the predicted answer. '
    'Check task relevance, a discriminator against the closest alternative, '
    'unresolved same-region conflicts, distinction between regional '
    'heterogeneity and contradiction, applicable scale/quantification, '
    'and the provenance of claimed assays. A plausible answer or a forced '
    'choice is not sufficient evidence. Return Yes only when the relevant '
    'checks pass; otherwise return No and identify the specific failed checks. '
    'Do not invent findings or claim access to earlier rounds absent here.'
)
REFLECT_SCHEMA = (
    'Output ONLY a JSON object with keys "sufficient" ("Yes" or "No"), '
    '"reason" (a brief evidence-grounded string), and "failed_checks" '
    '(a list of short check names, empty when sufficient is "Yes").'
)
MAGNIFICATION = (
    'Choose a concrete visual discriminator, then the action and scale. '
    'For missing representative tissue or a distributional question, prefer '
    'exploration to magnifying an irrelevant region. Zoom only when the '
    'current region contains a relevant target whose details may resolve '
    'the uncertainty. Begin with architecture, then cellular details. '
    'Physical-scale heuristics: consider 10 for architectural relationships '
    'needing broad context, 20 for local architecture with cellular detail, '
    'and 40 only for a decisive smaller cellular feature when the narrower '
    'field is justified. These are conditional starting points, not mandatory '
    'clinical thresholds. Choose the lowest allowed scale expected to be '
    'sufficient for this target, with one zoom available before Final; do not '
    'require a sequence of magnifications or always choose the highest. '
    'For relative scales, 2/4/8 are crop factors, not physical magnification '
    'or HPF; choose the smallest useful factor. A larger number does not '
    'guarantee new source-pixel detail. Zoom cannot obtain unprovided IHC, '
    'molecular tests, or calibrated counts. Describe such limitations honestly; '
    'do not claim an unavailable external-test or stopping action. '
    'Use a concrete observable noun phrase for missing_info, without assuming '
    'the desired answer. Explain the target and field/detail tradeoff in '
    'zoom_reason. When recommending zoom, recommended_zoom_level must be a '
    'JSON integer from the supplied allowed scales, never a quoted number, '
    'a decimal, or a value with x. Otherwise use the string "None".'
)
ZOOM_OBSERVATION = (
    'Inspect the requested feature in this crop while retaining local '
    'architectural context. State whether it is observed, not seen, or '
    'not assessable at the available detail. Do not turn a request for a '
    'marker into an observed stain or treat nominal scale as a calibrated '
    'field. A negative crop does not establish whole-slide absence.'
)
FINAL_REVIEW = (
    'Integrate all supplied regions and scales, including the latest zoom '
    'finding and its limitations. Do not discard contrary or unassessable '
    'new findings. Distinguish same-region contradictions from cross-region '
    'heterogeneity and repeated descriptions from independent regions. '
    'Check that the answer, explanation, task taxonomy, and cited findings '
    'agree. An answer may change with new evidence or correction of a '
    'specific earlier inference; identify that basis and the relevant region '
    'when possible. Do not mechanically preserve or freely replace the '
    'previous answer. When choices exist, return exactly one allowed label '
    'or option text. If the task forces a choice beyond the observed evidence, '
    'make the best supported uncertain indirect estimate and explicitly '
    'state its limitation in explanation; do not invent an assay, finding, '
    'threshold, or numerical measurement to make it appear certain. Keep '
    'the existing answer/explanation JSON schema.'
)
SUMMARY_PRESERVATION = (
    'Preserve decisive positive and negative observations, not-assessable '
    'statements, same-region conflicts, assay provenance, and scale '
    'limitations with their region identities. Keep distinct regions and '
    'zoom findings attributable. Do not promote speculation or scratch work '
    'to a confirmed observation, erase a conflicting latest finding, or '
    'convert missing evidence into absence.'
)


def _append(values, name, package, text):
    values[name] += '\n\n[' + RULE_VERSION + ':' + package + '] ' + text


def _replace_once(value, before, after, name):
    if value.count(before) != 1:
        raise ValueError('Unexpected base prompt schema in ' + name)
    return value.replace(before, after, 1)


def compile_variant(paper_values, context_values, variant):
    """Return two new constant dictionaries; never mutate the supplied base."""
    paper, context = dict(paper_values), dict(context_values)
    for package in selected_packages(variant):
        if package == 'O':
            _append(paper, 'PERCEPTOR_SYSTEM', package, OBSERVATION)
            _append(paper, 'GENERIC_PROMPT', package,
                    'Start the report with "Generic observation:".')
            _append(paper, 'QUESTION_PROMPT', package,
                    'Start the report with "Task observation:". Describe '
                    'question-relevant features without supplying unobserved facts.')
            _append(paper, 'PREDICT_SYSTEM', package, OBSERVATION_PREDICT)
        elif package == 'D':
            _append(paper, 'PREDICT_SYSTEM', package, DIFFERENTIAL)
        elif package == 'R':
            paper['REFLECT_SYSTEM'] = _replace_once(
                paper['REFLECT_SYSTEM'],
                'Output ONLY a JSON object: {"sufficient": "Yes" or "No"}.',
                REFLECT_SCHEMA, 'REFLECT_SYSTEM')
            _append(paper, 'REFLECT_SYSTEM', package, REFLECTION)
            paper['REFLECT_USER'] = _replace_once(
                paper['REFLECT_USER'],
                'Only return a JSON object: {{"sufficient": "Yes" or "No"}}',
                REFLECT_SCHEMA, 'REFLECT_USER')
        elif package == 'M':
            _append(paper, 'MISSING_SYSTEM', package, MAGNIFICATION)
            _append(context, 'ZOOM_MISSING_USER', package, ZOOM_OBSERVATION)
        elif package == 'F':
            _append(paper, 'FINAL_SYSTEM', package, FINAL_REVIEW)
            _append(paper, 'SUMMARY_SYSTEM', package, SUMMARY_PRESERVATION)
    return paper, context


def _literal_strings(source):
    values = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, str):
                values[node.targets[0].id] = value
    return values


def _render(source, before, after, variant):
    changed = {name: value for name, value in after.items()
               if value != before.get(name)}
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in changed):
            name = node.targets[0].id
            edits.append((node.lineno - 1, node.end_lineno,
                          name + ' = ' + repr(changed[name]) + '\n'))
    if len(edits) != len(changed):
        raise ValueError('Cannot locate every changed prompt assignment')
    for start, end, replacement in reversed(edits):
        lines[start:end] = [replacement]
    hashes = {name: hashlib.sha256(before[name].encode()).hexdigest()
              for name in sorted(changed)}
    # Wrap the manifest only: base source/publication metadata are captured
    # from the original manifest builder before attribution is overwritten.
    metadata = f'''

{_METADATA_MARKER}
_RULES_BASE_MANIFEST = prompt_manifest
_RULES_BASE_HASHES = {hashes!r}


def prompt_manifest():
    manifest = _RULES_BASE_MANIFEST()
    for name, base_hash in _RULES_BASE_HASHES.items():
        item = manifest['templates'][name]
        item['base_hash'] = base_hash
        item['base_source'] = item['source']
        item['base_published'] = item['published']
        item['published'] = False
        item['source'] = 'local rules-v0 prompt variant {variant}'
    manifest['variant'] = {variant!r}
    manifest['rules'] = {list(selected_packages(variant))!r}
    manifest['rule_version'] = {RULE_VERSION!r}
    return manifest
'''
    result = ''.join(lines).rstrip() + metadata
    ast.parse(result)
    return result


def render_variant_sources(paper_source, context_source, variant):
    """Return the two snapshot-relative prompt paths and their complete text."""
    selected_packages(variant)
    if any(_METADATA_MARKER in source for source in (paper_source, context_source)):
        raise ValueError('Variants must compile from an unmodified base snapshot')
    base_paper, base_context = map(_literal_strings, (paper_source, context_source))
    paper, context = compile_variant(base_paper, base_context, variant)
    return {
        'models/paper_prompts.py': _render(paper_source, base_paper, paper, variant),
        'models/context_prompts.py': _render(context_source, base_context, context, variant),
    }


def build_variant_sources(source_root, variant):
    root = Path(source_root)
    return render_variant_sources(
        (root / 'models/paper_prompts.py').read_text(encoding='utf-8'),
        (root / 'models/context_prompts.py').read_text(encoding='utf-8'), variant)
