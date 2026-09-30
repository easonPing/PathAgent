"""Protocol selection without loading models or depending on a machine's paths."""
import hashlib
from pathlib import Path

LEGACY_PROTOCOL = 'legacy-v1'
PAPER_PROTOCOL = 'paper-prompts-nonthinking-2x-v1'
CONTEXT_PROTOCOL = 'paper-prompts-context-restored-v3'
PAPER_UPSTREAM = '1e4d0377a4438b19d0e2dde6803db29f804ac8f4'


def protocol_name(config):
    name = config.get('protocol', LEGACY_PROTOCOL)
    if name not in (LEGACY_PROTOCOL, PAPER_PROTOCOL, CONTEXT_PROTOCOL):
        raise ValueError(f'Unknown inference protocol: {name}')
    return name


def backend_class(config):
    if protocol_name(config) == CONTEXT_PROTOCOL:
        from models.context_inference import ContextModelBackend
        return ContextModelBackend
    if protocol_name(config) == PAPER_PROTOCOL:
        from models.paper_inference import PaperModelBackend
        return PaperModelBackend
    from models.inference import ModelBackend
    return ModelBackend


def protocol_manifest(config):
    name = protocol_name(config)
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


def terminal_status(config, status):
    return status == 'ok' or (protocol_name(config) in (PAPER_PROTOCOL, CONTEXT_PROTOCOL) and status == 'invalid')
