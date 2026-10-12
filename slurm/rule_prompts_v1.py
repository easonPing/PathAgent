"""Short rules-v1 prompt compiler; model messages contain no experiment labels.

A changes the H&E prediction contract, B audits observation conflicts, and D
changes scale/exposure instructions. C is solely a controller toggle. The
existing v0 compiler is reused for AST substitution and remains untouched.
"""
import ast
import hashlib
from pathlib import Path

from slurm.rule_prompts import (
    _METADATA_MARKER as _V0_METADATA_MARKER,
    _literal_strings,
    _render as _render_v0,
    _replace_once,
)


CONTROL = 'BASE_COMPAT'
VARIANTS = ('A_HE_PRED', 'B_CONFLICT', 'C_ZOOM_LOOP', 'D_SCALE', 'ALL4')
RULE_VERSION = 'rules-v1'
PACKAGES = ('A', 'B', 'D')
_METADATA_MARKER = '# RULES_V1_MANIFEST_METADATA'
_PACKAGE_MAP = {
    CONTROL: (), 'A_HE_PRED': ('A',), 'B_CONFLICT': ('B',),
    'C_ZOOM_LOOP': (), 'D_SCALE': ('D',), 'ALL4': PACKAGES,
}
ZOOM_ENABLED = {variant: variant in ('C_ZOOM_LOOP', 'ALL4')
                for variant in (CONTROL,) + VARIANTS}

# The non-H&E clauses are the original baseline requirements, retained
# verbatim inside the applicable branch rather than weakened for every task.
BASE_REFLECT_TASK = (
    'Your task is to judge whether the current patch descriptions are '
    'sufficient to confidently support the answer.'
)
BASE_FINAL_REQUIREMENT = (
    'The answer should directly reflect the information observable in the '
    'slide, such as biomarker expression level, presence or absence of '
    'features, or a numeric measurement.'
)
A_PREDICT = (
    'For biomarker or molecular-label questions from H&E, predict the '
    'best-fitting option from morphology as an indirect estimate. Put the '
    'option in answer and the morphological basis and uncertainty in '
    'thinking_steps.'
)
A_REFLECT = (
    'For biomarker or molecular labels from H&E, judge an indirect '
    'morphological estimate without requiring an assay; for other questions, '
    'judge whether the current patch descriptions are sufficient to '
    'confidently support the answer.'
)
A_MISSING = (
    'For H&E biomarker or molecular estimates, express missing_info as the '
    'image-visible morphology needed to choose between the options.'
)
A_FINAL = (
    'For biomarker or molecular labels from H&E, return the best-fitting '
    'option as an indirect morphological estimate, with the option in answer '
    'and its basis and uncertainty in explanation. For other questions: '
    + BASE_FINAL_REQUIREMENT
)
B_PREDICT = (
    'Compare repeated descriptions of the same Region for incompatible '
    'observations of the same feature, allowing omissions and differences '
    'across regions. Briefly record any material conflict, the Region ID, '
    'and the two claims in thinking_steps.'
)
B_REFLECT = (
    'Return No when an unresolved same-region, same-feature contradiction '
    'affects the answer. An omission or a difference between regions is '
    'not itself a contradiction.'
)
B_MISSING = (
    'Express a material conflict as a neutral visual feature to inspect, '
    'without naming the diagnosis to confirm. Record the disputed feature '
    'and Region ID in zoom_reason.'
)
B_FINAL = (
    'Resolve recorded conflicts using observations of the relevant feature '
    'and region; otherwise state the remaining conflict in explanation.'
)
D_REFLECT = (
    'Use this scale map: 5x for distribution, localization and '
    'representativeness; 10x for glands, cell nests and large architecture; '
    '20x for cell cohesion, arrangement and nuclei; 40x for fine cellular '
    'features and mitoses. Return No when the answer requires a feature not '
    'resolved at the current scale or a more representative region.'
)
D_MISSING = (
    'For missing distribution, localization or representativeness, set '
    'zoom_recommendation to No to explore another 5x region. Otherwise '
    'choose 10x for glands, cell nests and large architecture (20x if '
    'needed), 20x for cell cohesion, arrangement and nuclei (40x if needed), '
    'or 40x for fine cellular features and mitoses.'
)


def selected_packages(variant):
    """Return actual prompt packages; C and BASE_COMPAT return an empty tuple."""
    if variant not in _PACKAGE_MAP:
        raise ValueError('Unknown rules-v1 variant: ' + str(variant))
    return _PACKAGE_MAP[variant]


packages = selected_packages


def zoom_enabled(variant):
    selected_packages(variant)  # Validate before exposing the control flag.
    return ZOOM_ENABLED[variant]


def _append(values, name, text):
    values[name] += '\n\n' + text


def compile_variant(paper_values, context_values, variant):
    """Pure value-level compilation; the context wrappers are never modified."""
    paper, context = dict(paper_values), dict(context_values)
    for package in selected_packages(variant):
        if package == 'A':
            _append(paper, 'PREDICT_SYSTEM', A_PREDICT)
            paper['REFLECT_SYSTEM'] = _replace_once(
                paper['REFLECT_SYSTEM'], BASE_REFLECT_TASK,
                A_REFLECT, 'REFLECT_SYSTEM')
            _append(paper, 'MISSING_SYSTEM', A_MISSING)
            paper['FINAL_SYSTEM'] = _replace_once(
                paper['FINAL_SYSTEM'], BASE_FINAL_REQUIREMENT,
                A_FINAL, 'FINAL_SYSTEM')
        elif package == 'B':
            for name, text in (
                ('PREDICT_SYSTEM', B_PREDICT), ('REFLECT_SYSTEM', B_REFLECT),
                ('MISSING_SYSTEM', B_MISSING), ('FINAL_SYSTEM', B_FINAL),
            ):
                _append(paper, name, text)
        elif package == 'D':
            _append(paper, 'REFLECT_SYSTEM', D_REFLECT)
            _append(paper, 'MISSING_SYSTEM', D_MISSING)
    return paper, context


def _render(source, before, after, variant):
    # Reuse the existing AST constant replacement implementation. Its v0
    # metadata is discarded before the source is returned or executed.
    provisional = _render_v0(source, before, after, 'ALL')
    body = provisional.split(_V0_METADATA_MARKER, 1)[0].rstrip()
    hashes = {name: hashlib.sha256(before[name].encode()).hexdigest()
              for name in sorted(before) if after[name] != before[name]}
    metadata = f'''


{_METADATA_MARKER}
_RULES_V1_BASE_MANIFEST = prompt_manifest
_RULES_V1_BASE_HASHES = {hashes!r}


def prompt_manifest():
    manifest = _RULES_V1_BASE_MANIFEST()
    for name, base_hash in _RULES_V1_BASE_HASHES.items():
        item = manifest['templates'][name]
        item['base_hash'] = base_hash
        item['base_source'] = item['source']
        item['base_published'] = item['published']
        item['published'] = False
        item['source'] = 'local rules-v1 prompt variant {variant}'
    manifest['variant'] = {variant!r}
    manifest['rules'] = {list(selected_packages(variant))!r}
    manifest['rule_version'] = {RULE_VERSION!r}
    manifest['post_zoom_reflect'] = {zoom_enabled(variant)!r}
    return manifest
'''
    result = body + metadata
    ast.parse(result)
    return result


def render_variant_sources(paper_source, context_source, variant):
    """Return two snapshot-relative paths mapped to complete generated source."""
    selected_packages(variant)
    if any(marker in source for marker in (_METADATA_MARKER, _V0_METADATA_MARKER)
           for source in (paper_source, context_source)):
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
