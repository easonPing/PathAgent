"""Frozen model backend; explicit generation settings and strict response schemas."""
import json
import time
from copy import deepcopy
from data_processing.common import resolve, sample_seed, seed_everything
from data_processing.datasets import answer_label
from data_processing.regions import ImageSource, Region
from models.agent import unit_rows

GENERIC_PROMPT = 'Please describe the pathology features in this image.'
QUESTION_PROMPT = 'Please describe the pathology features related to the question: {question} in this image.'

class ModelProtocolError(RuntimeError):
    pass

def parse_json_object(text):
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

def choices_text(choices):
    return '\nChoices:\n' + '\n'.join(f'{k}. {v}' for k, v in choices.items()) if choices else ''

class ModelBackend:
    def __init__(self, config):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration, CLIPModel, CLIPProcessor)
        if not torch.cuda.is_available():
            raise RuntimeError('Real inference requires an allocated GPU; no CPU fallback')
        self.config, self.records = config, []
        self.case_id, self.stage_counts = 'initialization', {}
        self._source = None
        torch.set_num_threads(config['runtime']['cpu_threads'])
        local = config['runtime']['local_files_only']
        executor = str(resolve(config['models']['executor']['path']))
        perceptor = str(resolve(config['models']['perceptor']['path']))
        navigator = str(resolve(config['models']['navigator']['path']))
        started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(executor, local_files_only=local)
        self.executor = AutoModelForCausalLM.from_pretrained(
            executor, local_files_only=local, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
        self.processor = AutoProcessor.from_pretrained(perceptor, local_files_only=local)
        self.perceptor = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            perceptor, local_files_only=local, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
        self.clip_processor = CLIPProcessor.from_pretrained(navigator, local_files_only=local)
        self.navigator = CLIPModel.from_pretrained(navigator, local_files_only=local).to('cuda:0').eval()
        if self.executor.config.model_type != 'qwen3' or self.perceptor.config.model_type != 'qwen2_5_vl':
            raise ValueError('Unexpected model architectures')
        self.load_seconds = time.monotonic() - started
        self.generation = config['generation']
        self.context_limit = int(self.executor.config.max_position_embeddings)
        self.resolved_generation = {
            'executor': self.executor.generation_config.to_dict(),
            'perceptor': self.perceptor.generation_config.to_dict(),
            'overrides': self.generation, 'image_processor': self.processor.image_processor.to_dict(),
            'gpu': torch.cuda.get_device_name(0),
            'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'context_limit': self.context_limit,
        }

    def set_case(self, case_id):
        self.case_id, self.stage_counts = case_id, {}

    def _seed(self, stage):
        index = self.stage_counts.get(stage, 0)
        self.stage_counts[stage] = index + 1
        seed = sample_seed(self.config['seed'], self.case_id, f'{stage}:{index}')
        seed_everything(seed)
        return seed

    def source(self, region):
        if self._source is None or self._source.path != str(resolve(region.image_path)):
            if self._source is not None:
                self._source.close()
            self._source = ImageSource(region.image_path, region.mpp)
        return self._source

    def close(self):
        if self._source is not None:
            self._source.close()
            self._source = None

    def encode_text(self, text):
        import torch
        inputs = self.clip_processor(text=[text], return_tensors='pt', max_length=77,
                                     padding='max_length', truncation=True).to('cuda:0')
        with torch.inference_mode():
            features = self.navigator.get_text_features(**inputs).float().cpu().numpy()
        return unit_rows(features)

    def encode_images(self, images):
        import torch
        import numpy as np
        outputs = []
        batch_size = self.config['runtime']['image_batch_size']
        for start in range(0, len(images), batch_size):
            inputs = self.clip_processor(images=images[start:start + batch_size], return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                outputs.append(self.navigator.get_image_features(**inputs).float().cpu().numpy())
        return unit_rows(np.concatenate(outputs))

    def encode_regions(self, regions):
        import numpy as np
        output = []
        batch_size = self.config['runtime']['image_batch_size']
        for start in range(0, len(regions), batch_size):
            images = []
            for region in regions[start:start + batch_size]:
                images.append(self.source(region).read(region))
            output.append(self.encode_images(images))
            for image in images:
                image.close()
        return np.concatenate(output)

    def describe(self, region, question=None, missing_info=None):
        image = self.source(region).read(region)
        try:
            return self.describe_image(image, region.metadata(), question, missing_info)
        finally:
            image.close()

    def describe_image(self, image, metadata='', question=None, missing_info=None):
        import torch
        from qwen_vl_utils import process_vision_info
        prompt = GENERIC_PROMPT if question is None else QUESTION_PROMPT.format(question=question)
        if missing_info:
            prompt += f'\nMissing visual information to inspect: {missing_info}'
        messages = [
            {'role': 'system', 'content': 'You are an AI assistant specialized in describing visible pathology features.'},
            {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                         {'type': 'text', 'text': metadata + '\n' + prompt}]}]
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
                             'seed': seed, 'prompt': prompt, 'region_metadata': metadata, 'raw_response': output})
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
                             'raw_response': output, 'seed': seed, 'messages': messages})
        return output

    def _json(self, system, prompt, tokens, stage, keys, greedy=False, validator=None):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
        for attempt in range(self.generation['retries'] + 1):
            raw = self._text(messages, tokens, stage, greedy)
            parsed = parse_json_object(raw)
            valid = parsed is not None and all(k in parsed for k in keys)
            if valid and (validator is None or validator(parsed)):
                return parsed
            messages = deepcopy(messages[:2])
            messages.append({'role': 'user', 'content': 'Return one valid JSON object with the required keys and valid field values only.'})
        raise ModelProtocolError(f'Invalid {stage} response after {attempt + 1} attempts')

    def evidence_text(self, findings, question, extra=''):
        chunks = [Region(**f['region']).metadata() + '\n' + f['description'] for f in findings]
        algorithm = self.config['algorithm']
        budget = self.context_limit - self.generation['final_tokens'] - len(self.tokenizer.encode(extra + question)) - 2048
        if budget < 1024:
            raise ModelProtocolError('Reasoning history alone exceeds context')
        for level in range(5):
            text = '\n\n'.join(chunks)
            if len(self.tokenizer.encode(text)) <= budget and (level > 0 or len(chunks) <= algorithm['summary_threshold']):
                return text
            previous_tokens = len(self.tokenizer.encode(text))
            summarized = []
            for start in range(0, len(chunks), algorithm['summary_chunk_size']):
                group = chunks[start:start + algorithm['summary_chunk_size']]
                summary = self._text([
                    {'role': 'system', 'content': "Summarize visible pathological findings, preserving region coordinates and each region's scale. Do not give a final answer or diagnosis."},
                    {'role': 'user', 'content': f'Question: {question}\n' + '\n\n'.join(group)}],
                    self.generation['summary_tokens'], 'summary', greedy=True)
                summarized.append(summary)
            chunks = summarized
            if len(self.tokenizer.encode('\n\n'.join(chunks))) >= previous_tokens:
                raise ModelProtocolError('Summary did not reduce overlong evidence')
        raise ModelProtocolError('Unable to fit evidence without dropping observations')

    def evaluate(self, findings, sample, scale_kind, allowed_scales):
        description = self.evidence_text(findings, sample['question'])
        context = f"Descriptions:\n{description}\nQuestion: {sample['question']}{choices_text(sample['choices'])}"
        a = self._json(
            'You are an expert AI pathology assistant. Based on the patch descriptions, try to answer the question step-by-step. Output ONLY JSON: {"answer": "predicted answer", "thinking_steps": "reasoning"}.',
            context, self.generation['step_a_tokens'], 'predict', ['answer', 'thinking_steps'],
            validator=lambda x: isinstance(x['answer'], str) and isinstance(x['thinking_steps'], str))
        b = self._json(
            'Judge whether the current patch descriptions are sufficient to confidently support the answer. Output ONLY JSON: {"sufficient": "Yes" or "No"}.',
            context + '\nPrevious answer and reasoning:\n' + json.dumps(a),
            self.generation['step_b_tokens'], 'reflect', ['sufficient'],
            validator=lambda x: str(x['sufficient']).lower() in {'yes', 'no'})
        if b['sufficient'].lower() == 'yes':
            return {**a, **b}
        scale_description = 'absolute magnification' if scale_kind == 'physical' else 'relative crop factor; physical magnification is unknown'
        c = self._json(
            'Specify missing visual evidence and whether zooming helps. Output ONLY JSON with keys missing_info (short noun phrase), zoom_recommendation (Yes or No), zoom_level (number or null), zoom_reason (string).',
            context + '\nPrevious answer and reflection:\n' + json.dumps({**a, **b}) +
            f'\nScale means {scale_description}. Available zoom levels: {allowed_scales}.',
            self.generation['step_c_tokens'], 'missing_info',
            ['missing_info', 'zoom_recommendation', 'zoom_level', 'zoom_reason'],
            validator=lambda x: isinstance(x['missing_info'], str) and bool(x['missing_info'].strip())
            and str(x['zoom_recommendation']).lower() in {'yes', 'no'}
            and (str(x['zoom_recommendation']).lower() == 'no' or x['zoom_level'] in allowed_scales))
        return {**a, **b, **c}

    def conclude(self, findings, trajectory, sample):
        history = json.dumps([{k: e[k] for k in ('iteration', 'action', 'decision')} for e in trajectory], ensure_ascii=False)
        description = self.evidence_text(findings, sample['question'], history)
        prompt = (f"Question: {sample['question']}{choices_text(sample['choices'])}\n"
                  f'Accumulated region evidence:\n{description}\nPrior inferences and actions:\n{history}')
        return self._json(
            'You are an expert slide-level pathology assistant. Integrate all visual evidence and prior inferences to answer the specific question. Keep answer short. If choices are given, return exactly one option text or its label. Output ONLY JSON with answer and explanation (both strings).',
            prompt, self.generation['final_tokens'], 'final', ['answer', 'explanation'], greedy=True,
            validator=lambda x: isinstance(x['answer'], str) and bool(x['answer'].strip())
            and isinstance(x['explanation'], str) and (not sample['choices'] or answer_label(x['answer'], sample['choices']) is not None))
