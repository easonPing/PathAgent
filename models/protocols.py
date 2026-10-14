"""Protocol selection without loading models or depending on a machine's paths."""
import hashlib
from pathlib import Path

LEGACY_PROTOCOL = 'legacy-v1'
PAPER_PROTOCOL = 'paper-prompts-nonthinking-2x-v1'
CONTEXT_PROTOCOL = 'paper-prompts-context-restored-v3'
RULES_V2_PROTOCOL = 'pathology-rules-v2'
PAPER_UPSTREAM = '1e4d0377a4438b19d0e2dde6803db29f804ac8f4'


def protocol_name(config):
    name = config.get('protocol', LEGACY_PROTOCOL)
    if name not in (LEGACY_PROTOCOL, PAPER_PROTOCOL, CONTEXT_PROTOCOL, RULES_V2_PROTOCOL):
        raise ValueError(f'Unknown inference protocol: {name}')
    return name


def backend_class(config):
    if protocol_name(config) == RULES_V2_PROTOCOL:
        from models.rules_v2_inference import RulesV2ModelBackend
        return RulesV2ModelBackend
    if protocol_name(config) == CONTEXT_PROTOCOL:
        from models.context_inference import ContextModelBackend
        return ContextModelBackend
    if protocol_name(config) == PAPER_PROTOCOL:
        from models.paper_inference import PaperModelBackend
        return PaperModelBackend
    from models.inference import ModelBackend
    return ModelBackend


def _base_protocol_manifest(config):
    name = protocol_name(config)
    if name == RULES_V2_PROTOCOL:
        from models.rules_v2_prompts import prompt_manifest
        return {**prompt_manifest(config), 'upstream_commit': PAPER_UPSTREAM,
                'perceptor_sampling': 'checkpoint',
                'adaptations': ['Shared top-five Zoom and post-Zoom Predict/Reflect.',
                                'Opt-in rules R1 through R5; archived baseline is reused.']}
    if name == CONTEXT_PROTOCOL:
        from models.context_prompts import prompt_manifest
        return {**prompt_manifest(), 'upstream_commit': PAPER_UPSTREAM,
                'perceptor_sampling': 'checkpoint',
                'adaptations': ['Perceptor inherits checkpoint sampling, as in legacy-v1.',
                                'Local single-GPU BF16/FP32 backend and Trident environment.',
                                'Restore region metadata, Zoom missing_info, and current Predict/Reflect outputs.']}
    if name == PAPER_PROTOCOL:
        from models.paper_prompts import prompt_manifest
        return {**prompt_manifest(), 'upstream_commit': PAPER_UPSTREAM,
                'perceptor_sampling': 'checkpoint',
                'adaptations': ['Perceptor inherits checkpoint sampling, as in legacy-v1.',
                                'Local single-GPU BF16/FP32 backend and Trident environment.']}
    # Legacy prompts remain inline to preserve the original implementation.
    path = Path(__file__).with_name('inference.py')
    return {'protocol': name, 'perceptor_sampling': 'checkpoint',
            'prompt_source': 'models/inference.py',
            'prompt_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def protocol_manifest(config):
    from models.region_selection import selection_manifest, selection_policy
    manifest = _base_protocol_manifest(config)
    if selection_policy(config.get('algorithm', {})) == 'max5':
        manifest['region_selection'] = selection_manifest(config['algorithm'])
        if 'adaptations' in manifest:
            manifest['adaptations'] = [text.replace('Shared top-five Zoom', 'All available Zoom children')
                                       for text in manifest['adaptations']]
        if 'notes' in manifest:
            manifest['notes'] = [text.replace('batch Zoom top-5', 'all available Zoom children')
                                 for text in manifest['notes']]
    return manifest


def terminal_status(config, status):
    return status == 'ok' or (protocol_name(config) in (PAPER_PROTOCOL, CONTEXT_PROTOCOL, RULES_V2_PROTOCOL) and status == 'invalid')
