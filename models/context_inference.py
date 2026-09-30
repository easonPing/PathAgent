"""V3 restores context without changing v2 generation, parsing or agent control flow."""
import json

from models.paper_inference import PaperModelBackend
from models.context_prompts import PERCEPTOR_USER, ZOOM_MISSING_USER, MISSING_USER


class ContextModelBackend(PaperModelBackend):
    def perceptor_user(self, metadata='', question=None, missing_info=None):
        instruction = super().perceptor_user(metadata, question, missing_info)
        prompt = PERCEPTOR_USER.format(metadata=metadata, instruction=instruction)
        if missing_info:
            prompt += ZOOM_MISSING_USER.format(missing_info=missing_info)
        return prompt

    def missing_user(self, description, fields, scale_description, allowed_scales, prediction, reflection):
        return MISSING_USER.format(
            description=description, **fields,
            prediction=json.dumps({k: prediction[k] for k in ('answer', 'thinking_steps')}, ensure_ascii=False),
            reflection=json.dumps(reflection, ensure_ascii=False),
            scale_description=scale_description, allowed_scales=allowed_scales,
        )
