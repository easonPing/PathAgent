"""V3 input wrappers; published systems and v2 prompt snapshots remain unchanged."""
import hashlib

from models.paper_prompts import prompt_manifest as paper_manifest
from models.protocols import CONTEXT_PROTOCOL

PERCEPTOR_USER = '{metadata}\n{instruction}'
ZOOM_MISSING_USER = '\nMissing visual information to inspect: {missing_info}'
MISSING_USER = (
    'Descriptions:\n{description}\n\nQuestion: {question}{choices}\n\n'
    'Previous answer and reasoning:\n{prediction}\n\n'
    'Previous reflection:\n{reflection}\n\n'
    'Now provide the JSON for missing info and zoom recommendation.\n'
    'Scale means {scale_description}. Available zoom levels: {allowed_scales}.'
)


def prompt_manifest():
    manifest = paper_manifest()
    manifest['protocol'] = CONTEXT_PROTOCOL
    for name in ('PERCEPTOR_USER', 'ZOOM_MISSING_USER', 'MISSING_USER'):
        text = globals()[name]
        manifest['templates'][name] = {
            'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'published': False,
            'source': 'local v3 wrapper; prior-output information flow follows PathAgent supplement Prompt section',
        }
    manifest['notes'] += [
        'All Perceptor user messages include Region.metadata(); nonempty Zoom missing_info is appended verbatim.',
        'Missing receives current Predict answer/thinking_steps and the actual Reflect JSON; no reflection rationale is invented.',
        'Inherited v2 evidence budgeting includes full v3 wrappers and format retries.',
    ]
    return manifest
