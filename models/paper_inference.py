"""Paper protocol adapted from 1e4d037; Perceptor sampling follows the local baseline.

Only message/generation handling differs from the legacy backend. Model loading,
precision, device placement, image reading and embeddings use the local backend.
"""
import hashlib
import json
import time
from copy import deepcopy

from data_processing.regions import Region
from data_processing.datasets import answer_label
from models.inference import ModelBackend, ModelProtocolError, choices_text
from models.paper_prompts import (
    GENERIC_PROMPT, QUESTION_PROMPT, PERCEPTOR_SYSTEM, PREDICT_SYSTEM,
    REFLECT_SYSTEM, MISSING_SYSTEM, FINAL_SYSTEM, SUMMARY_SYSTEM,
    PREDICT_USER, REFLECT_USER, MISSING_USER, FINAL_USER, RETRY_USER,
)

def answer_content(text):
    """Non-thinking output needs no marker; never parse an explicit thought block."""
    while '<think>' in text:
        start = text.index('<think>')
        end = text.find('</think>', start)
        if end < 0:
            return ''
        text = text[:start] + text[end + len('</think>'):]
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    return text.strip()


def parse_json_object(text):
    text = answer_content(text)
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    return None


class PaperModelBackend(ModelBackend):
    def perceptor_user(self, metadata='', question=None, missing_info=None):
        return GENERIC_PROMPT if question is None else QUESTION_PROMPT.format(question=question)

    def missing_user(self, description, fields, scale_description, allowed_scales, prediction, reflection):
        return MISSING_USER.format(description=description, scale_description=scale_description,
                                   allowed_scales=allowed_scales, **fields)

    def describe(self, region, question=None, missing_info=None):
        image = self.source(region).read(region)
        try:
            description = self.describe_image(image, region.metadata(), question, missing_info)
            self.records[-1]['region'] = region.to_dict()
            return description
        finally:
            image.close()


    def describe_image(self, image, metadata='', question=None, missing_info=None):
        import torch
        from qwen_vl_utils import process_vision_info
        prompt = self.perceptor_user(metadata, question, missing_info)
        messages = [
            {'role': 'system', 'content': PERCEPTOR_SYSTEM},
            {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                         {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors='pt', padding=True).to('cuda:0')
        tokens = self.generation['generic_description_tokens' if question is None else 'question_description_tokens']
        seed = self._seed('describe:' + metadata + str(question) + str(missing_info))
        started = time.monotonic()
        with torch.inference_mode():
            generated = self.perceptor.generate(**inputs, max_new_tokens=tokens)
        output = self.processor.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        self.records.append({'stage': 'describe', 'seconds': time.monotonic() - started,
                             'input_tokens': inputs.input_ids.shape[1], 'output_tokens': generated.shape[1] - inputs.input_ids.shape[1],
                             'seed': seed, 'prompt': prompt, 'region_metadata': metadata, 'raw_response': output,
                             'answer_content': output, 'generation': dict(self.perceptor.generation_config.to_dict(), max_new_tokens=tokens),
                             'at_token_cap': generated.shape[1] - inputs.input_ids.shape[1] >= tokens,
                             'finish_reason': 'eos' if int(generated[0, -1]) == self.perceptor.generation_config.eos_token_id else 'length_or_stop',
                             'input_size': list(image.size), 'processed_sizes': [list(im.size) for im in image_inputs],
                             'image_grid_thw': inputs.image_grid_thw.tolist(),
                             'visual_tokens': int(inputs.image_grid_thw.prod(-1).sum()) // self.processor.image_processor.merge_size ** 2,
                             'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                             'messages': [{'role': 'system', 'content': PERCEPTOR_SYSTEM},
                                          {'role': 'user', 'content': [{'type': 'image', 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'metadata': metadata},
                                                                       {'type': 'text', 'text': prompt}]}]})
        if not output:
            raise ModelProtocolError('Perceptor returned an empty description')
        return output


    def _text(self, messages, tokens, stage, greedy=False):
        import torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.generation['enable_thinking'])
        inputs = self.tokenizer([text], return_tensors='pt').to('cuda:0')
        if inputs.input_ids.shape[1] + tokens > self.context_limit:
            raise ModelProtocolError(f'Context overflow at {stage}: {inputs.input_ids.shape[1]} + {tokens}')
        kwargs = {'max_new_tokens': tokens}
        if greedy:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        started = time.monotonic()
        seed = self._seed(stage)
        with torch.inference_mode():
            generated = self.executor.generate(**inputs, **kwargs)
        output = self.tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        self.records.append({'stage': stage, 'seconds': time.monotonic() - started,
                             'input_tokens': inputs.input_ids.shape[1], 'output_tokens': generated.shape[1] - inputs.input_ids.shape[1],
                             'raw_response': output, 'answer_content': answer_content(output), 'seed': seed, 'messages': messages,
                             'generation': dict(self.executor.generation_config.to_dict(), **kwargs),
                             'enable_thinking': self.generation['enable_thinking'],
                             'at_token_cap': generated.shape[1] - inputs.input_ids.shape[1] >= tokens,
                             'finish_reason': 'eos' if int(generated[0, -1]) in ([self.executor.generation_config.eos_token_id] if isinstance(self.executor.generation_config.eos_token_id, int) else self.executor.generation_config.eos_token_id) else 'length_or_stop'})
        return answer_content(output)


    def _json(self, system, prompt, tokens, stage, keys, greedy=False, validator=None):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
        for attempt in range(self.generation['retries'] + 1):
            raw = self._text(messages, tokens, stage, greedy)
            parsed = parse_json_object(raw)
            valid = parsed is not None and all(k in parsed for k in keys)
            if valid and (validator is None or validator(parsed)):
                return parsed
            messages = deepcopy(messages[:2])
            messages.append({'role': 'user', 'content': RETRY_USER})
        raise ModelProtocolError(f'Invalid {stage} response after {attempt + 1} attempts')


    def evidence_text(self, findings, question, extra='', requests=None):
        chunks = [Region(**f['region']).metadata() + '\n' + f['description'] for f in findings]
        algorithm = self.config['algorithm']
        budget = self.context_limit - self.generation['final_tokens'] - len(self.tokenizer.encode(extra + question)) - 2048
        if budget < 1024:
            raise ModelProtocolError('Reasoning history alone exceeds context')
        for level in range(5):
            text = '\n\n'.join(chunks)
            fits = len(self.tokenizer.encode(text)) <= budget
            if requests is not None:
                fits = all(len(self.tokenizer.encode(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))) + limit <= self.context_limit
                    for messages, limit in requests(text))
            if fits and (level > 0 or len(chunks) <= algorithm['summary_threshold']):
                return text
            previous_tokens = len(self.tokenizer.encode(text))
            summarized = []
            for start in range(0, len(chunks), algorithm['summary_chunk_size']):
                group = chunks[start:start + algorithm['summary_chunk_size']]
                summary = self._text([
                    {'role': 'system', 'content': SUMMARY_SYSTEM},
                    {'role': 'user', 'content': f'Question: {question}\n' + '\n\n'.join(group)}],
                    self.generation['summary_tokens'], 'summary', greedy=True)
                summarized.append(summary)
            chunks = summarized
            if len(self.tokenizer.encode('\n\n'.join(chunks))) >= previous_tokens:
                raise ModelProtocolError('Summary did not reduce overlong evidence')
        raise ModelProtocolError('Unable to fit evidence without dropping observations')


    def _evidence_for(self, findings, system, render, tokens, question=""):
        def requests(description):
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': render(description)}]
            # Include the format retry envelope in the context budget.
            return [(messages + [{'role': 'user', 'content': RETRY_USER}], tokens)]
        return self.evidence_text(findings, question, requests=requests)


    def evaluate(self, findings, sample, scale_kind, allowed_scales):
        fields = {'question': sample['question'], 'choices': choices_text(sample['choices'])}
        render = lambda description: PREDICT_USER.format(description=description, **fields)
        description = self._evidence_for(findings, PREDICT_SYSTEM, render, self.generation['step_a_tokens'], sample['question'])
        a = self._json(PREDICT_SYSTEM, render(description), self.generation['step_a_tokens'],
                       'predict', ['answer', 'thinking_steps'],
                       validator=lambda x: isinstance(x['answer'], str) and isinstance(x['thinking_steps'], str))
        render = lambda description: REFLECT_USER.format(description=description, answer=json.dumps(a, ensure_ascii=False), **fields)
        description = self._evidence_for(findings, REFLECT_SYSTEM, render, self.generation['step_b_tokens'], sample['question'])
        b = self._json(REFLECT_SYSTEM, render(description), self.generation['step_b_tokens'],
                       'reflect', ['sufficient'], validator=lambda x: str(x['sufficient']).lower() in {'yes', 'no'})
        if b['sufficient'].lower() == 'yes':
            return {**a, **b}
        scale_description = 'absolute magnification' if scale_kind == 'physical' else 'relative crop factor; physical magnification is unknown'
        render = lambda description: self.missing_user(description, fields, scale_description,
                                                       allowed_scales, a, b)
        description = self._evidence_for(findings, MISSING_SYSTEM, render, self.generation['step_c_tokens'], sample['question'])
        def valid_missing(x):
            level = x['recommended_zoom_level']
            if isinstance(level, str) and level in ('10', '20', '40'):
                level = x['recommended_zoom_level'] = int(level)
            if level == 'None':
                level = None
            return (isinstance(x['missing_info'], str) and bool(x['missing_info'].strip())
                    and isinstance(x['zoom_reason'], str)
                    and str(x['zoom_recommendation']).lower() in {'yes', 'no'}
                    and (level is None or (type(level) is int and level in allowed_scales))
                    and (str(x['zoom_recommendation']).lower() == 'no' or level in allowed_scales))
        c = self._json(MISSING_SYSTEM, render(description), self.generation['step_c_tokens'], 'missing_info',
                       ['missing_info', 'zoom_recommendation', 'recommended_zoom_level', 'zoom_reason'], validator=valid_missing)
        c['zoom_level'] = None if c['recommended_zoom_level'] == 'None' else c['recommended_zoom_level']
        return {**a, **b, **c}


    def conclude(self, findings, trajectory, sample):
        history = json.dumps([{k: e[k] for k in ('iteration', 'action', 'decision')} for e in trajectory], ensure_ascii=False)
        render = lambda description: FINAL_USER.format(question=sample['question'], choices=choices_text(sample['choices']),
                                                       description=description, history=history)
        description = self._evidence_for(findings, FINAL_SYSTEM, render, self.generation['final_tokens'], sample['question'])
        return self._json(FINAL_SYSTEM, render(description), self.generation['final_tokens'],
                         'final', ['answer', 'explanation'], greedy=True,
                         validator=lambda x: isinstance(x['answer'], str) and bool(x['answer'].strip())
                         and isinstance(x['explanation'], str)
                         and (not sample['choices'] or answer_label(x['answer'], sample['choices']) is not None))
