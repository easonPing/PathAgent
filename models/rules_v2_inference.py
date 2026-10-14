"""Opt-in rules-v2 messages and deterministic evidence checks.

The archived ContextModelBackend remains unchanged. Model loading, numerical
precision, generation settings and stage-derived seeds are inherited verbatim.
"""
import hashlib
import json
import os
import re
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from data_processing.datasets import answer_label
from data_processing.regions import Region
from models.context_inference import ContextModelBackend
from models.inference import ModelProtocolError, choices_text
from models.paper_inference import answer_content, parse_json_object
from models.rules_v2_prompts import (
    REQUIRED_SCALES, flags_for_variant, missing_keys, predict_keys, reflect_keys,
    schemas_for_flags, templates_for_config,
)


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def normalize_tissue_response(value):
    """Normalize two known protocol keys, without inferring tissue state.

    Case, spaces and underscores in keys have no clinical meaning. Ambiguous
    aliases, missing fields and invalid values still require a format retry.
    """
    if not isinstance(value, dict):
        return None, {}
    fields, renamed = {}, {}
    known = {'tissuestatus': 'tissue_status', 'findings': 'findings'}
    for key, item in value.items():
        canonical = known.get(re.sub(r'[\s_]+', '', key).lower())
        if canonical is None:
            continue
        if canonical in fields and fields[canonical] != item:
            return None, renamed
        fields[canonical] = item
        if key != canonical:
            renamed[key] = canonical
    status, findings = fields.get('tissue_status'), fields.get('findings')
    if not isinstance(status, str) or status not in {'present', 'absent', 'uncertain'} or not _nonempty(findings):
        return None, renamed
    return fields, renamed


def _repair_findings_quotes(content):
    """Repair only bare quotes in a complete, two-known-string-field object."""
    body = content.strip()
    wrapper = re.fullmatch(r'<answer>\s*(.*?)\s*</answer>', body, flags=re.S | re.I)
    if wrapper:
        body = wrapper.group(1).strip()
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', body, flags=re.S | re.I)
    if fence:
        body = fence.group(1).strip()
    patterns = (
        r'\{\s*"(?P<status_key>[^"\\]+)"\s*:\s*"(?P<status>present|absent|uncertain)"\s*,\s*"(?P<findings_key>[^"\\]+)"\s*:\s*"(?P<findings>.*)"\s*\}',
        r'\{\s*"(?P<findings_key>[^"\\]+)"\s*:\s*"(?P<findings>.*)"\s*,\s*"(?P<status_key>[^"\\]+)"\s*:\s*"(?P<status>present|absent|uncertain)"\s*\}',
    )
    match = next((m for pattern in patterns for m in [re.fullmatch(pattern, body, flags=re.S)] if m), None)
    if match is None:
        return None, None
    known = {'tissuestatus': 'tissue_status', 'findings': 'findings'}
    status_key, findings_key = match.group('status_key'), match.group('findings_key')
    if (known.get(re.sub(r'[\s_]+', '', status_key).lower()) != 'tissue_status'
            or known.get(re.sub(r'[\s_]+', '', findings_key).lower()) != 'findings'):
        return None, None
    findings = match.group('findings')
    # A second/extra field may not be swallowed by the greedy findings match.
    if re.search(r'(?<!\\)"\s*,\s*"[^"\\]+"\s*:', findings):
        return None, None
    escaped, inserted = [], 0
    for i, char in enumerate(findings):
        slashes, j = 0, i - 1
        while j >= 0 and findings[j] == '\\':
            slashes += 1
            j -= 1
        if char == '"' and slashes % 2 == 0:
            escaped.append('\\')
            inserted += 1
        escaped.append(char)
    if not inserted:
        return None, None
    repaired = ('{' + json.dumps(status_key) + ':' + json.dumps(match.group('status'))
                + ',' + json.dumps(findings_key) + ':"' + ''.join(escaped) + '"}')
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None, None
    return parsed, {'method': 'escape_unescaped_quotes_in_findings',
                    'inserted_backslashes': inserted, 'raw_container': body,
                    'repaired_json': repaired}


def parse_tissue_response(text, return_repair=False):
    """Use explicit states only; one bounded JSON quoting repair is auditable."""
    content = answer_content(text)
    def finish(parsed, renamed, wire, repair=None):
        values = (parsed, renamed, wire)
        return values + (repair,) if return_repair else values
    def strict_json():
        start = content.find('{')
        if start < 0:
            return None
        def unique_pairs(pairs):
            result, known = {}, set()
            aliases = {'tissuestatus': 'tissue_status', 'findings': 'findings'}
            for key, item in pairs:
                canonical = aliases.get(re.sub(r'[\s_]+', '', key).lower())
                if key in result or (canonical is not None and canonical in known):
                    raise ValueError('Duplicate protocol field')
                result[key] = item
                if canonical is not None:
                    known.add(canonical)
            return result
        try:
            value, _ = json.JSONDecoder(object_pairs_hook=unique_pairs).raw_decode(content[start:])
        except (json.JSONDecodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    json_start = re.match(r'^\s*(?:<answer>\s*)?(?:```(?:json)?\s*)?\{', content, flags=re.I)
    if json_start:
        parsed, renamed = normalize_tissue_response(strict_json())
        if parsed is not None:
            return finish(parsed, renamed, 'json')
        value, repair = _repair_findings_quotes(content)
        parsed, renamed = normalize_tissue_response(value)
        return finish(parsed, renamed, 'json', repair if parsed is not None else None)
    statuses = re.findall(r'<tissue_status>\s*(.*?)\s*</tissue_status>', content, flags=re.S | re.I)
    observations = re.findall(r'<findings>\s*(.*?)\s*</findings>', content, flags=re.S | re.I)
    if statuses or observations:
        if len(statuses) != 1 or len(observations) != 1:
            return finish(None, {}, 'tags')
        value = {'tissue_status': statuses[0].strip(), 'findings': observations[0].strip()}
        parsed, renamed = normalize_tissue_response(value)
        return finish(parsed, renamed, 'tags')
    parsed, renamed = normalize_tissue_response(strict_json())
    if parsed is not None:
        return finish(parsed, renamed, 'json')
    value, repair = _repair_findings_quotes(content)
    parsed, renamed = normalize_tissue_response(value)
    return finish(parsed, renamed, 'json', repair if parsed is not None else None)


class RulesV2ModelBackend(ContextModelBackend):
    """Rules are selected only by rules_v2.variant; there are no global defaults."""

    def __init__(self, config):
        super().__init__(config)
        self.rule_flags = flags_for_variant(config['rules_v2']['variant'])
        self.rule_templates = templates_for_config(config)
        self.tissue_status = {}
        self.tissue_status_source = {}
        self.evidence_artifacts = {}

    def set_case(self, case_id):
        super().set_case(case_id)
        self.tissue_status = {}
        self.tissue_status_source = {}
        self.evidence_artifacts = {}

    def describe(self, region, question=None, missing_info=None):
        description = super().describe(region, question, missing_info)
        record = self.records[-1]
        if record.get('image_artifact_path'):
            self.evidence_artifacts[region.region_id] = {
                key: deepcopy(record[key]) for key in
                ('image_artifact_path', 'pixel_sha256', 'input_size', 'input_mode')}
        return description

    def perceptor_user(self, metadata='', question=None, missing_info=None):
        templates = self.rule_templates
        instruction = (templates['GENERIC_PROMPT'] if question is None else
                       templates['QUESTION_PROMPT'].format(question=question))
        prompt = templates['PERCEPTOR_USER'].format(metadata=metadata, instruction=instruction)
        if 'R2' in self.rule_flags:
            prompt += '\nInput modality: H&E'
        if missing_info:
            prompt += templates['ZOOM_MISSING_USER'].format(missing_info=missing_info)
        if question is not None and 'R3' in self.rule_flags:
            prompt += '\n\n' + templates['QUESTION_OUTPUT_USER']
        return prompt

    def perceptor_system(self, question=None):
        system = self.rule_templates['PERCEPTOR_SYSTEM']
        if question is not None and 'R3' in self.rule_flags:
            system += '\n\n' + self.rule_templates['QUESTION_OUTPUT_SYSTEM']
        return system

    def save_evidence_image(self, image):
        """Persist the exact model-input pixels in the new campaign cache only."""
        root = self.config.get('runtime', {}).get('rules_v2_evidence_image_root')
        if root is None:
            return None
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        directory = Path(root).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (digest + '.png')
        if not target.exists():
            handle, temporary = tempfile.mkstemp(prefix=digest + '.', suffix='.png', dir=str(directory))
            os.close(handle)
            try:
                image.save(temporary, format='PNG')
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return str(target)

    def describe_image(self, image, metadata='', question=None, missing_info=None):
        """Use the effective Perceptor system, retaining baseline generation calls.

        R3 validates the non-thinking JSON and allows one format retry. It never
        interprets a malformed response as absence of tissue.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = self.perceptor_user(metadata, question, missing_info)
        system = self.perceptor_system(question)
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                         {'type': 'text', 'text': prompt}]}]
        parse_tissue = question is not None and 'R3' in self.rule_flags
        tokens = self.generation['generic_description_tokens' if question is None
                                 else 'question_description_tokens']
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
        image_artifact = self.save_evidence_image(image) if question is not None else None
        stage_seed = 'describe:' + metadata + str(question) + str(missing_info)
        for attempt in range(2 if parse_tissue else 1):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, return_tensors='pt', padding=True).to('cuda:0')
            seed = self._seed(stage_seed)
            started = time.monotonic()
            with torch.inference_mode():
                generated = self.perceptor.generate(**inputs, max_new_tokens=tokens)
            output = self.processor.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            output_length = generated.shape[1] - inputs.input_ids.shape[1]
            eos = self.perceptor.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            recorded_messages = [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': [
                    {'type': 'image', 'pixel_sha256': pixel_sha, 'metadata': metadata},
                    {'type': 'text', 'text': prompt}]}]
            recorded_messages.extend(deepcopy(messages[2:]))
            self.records.append({
                'stage': 'describe', 'seconds': time.monotonic() - started,
                'input_tokens': inputs.input_ids.shape[1], 'output_tokens': output_length,
                'seed': seed, 'prompt': prompt, 'region_metadata': metadata,
                'raw_response': output, 'answer_content': answer_content(output),
                'generation': dict(self.perceptor.generation_config.to_dict(), max_new_tokens=tokens),
                'at_token_cap': output_length >= tokens,
                'finish_reason': 'eos' if int(generated[0, -1]) in eos else 'length_or_stop',
                'input_size': list(image.size), 'processed_sizes': [list(im.size) for im in image_inputs],
                'image_grid_thw': inputs.image_grid_thw.tolist(),
                'visual_tokens': int(inputs.image_grid_thw.prod(-1).sum()) // self.processor.image_processor.merge_size ** 2,
                'pixel_sha256': pixel_sha, 'messages': recorded_messages,
                'format_attempt': attempt,
                'image_artifact_path': image_artifact, 'input_mode': image.mode,
            })
            if not parse_tissue:
                if not output:
                    raise ModelProtocolError('Perceptor returned an empty description')
                return output
            parsed, renamed, wire_format, repair = parse_tissue_response(output, return_repair=True)
            self.records[-1]['key_normalization'] = renamed
            self.records[-1]['question_wire_format'] = wire_format
            if repair is not None:
                self.records[-1]['syntax_repair'] = repair
            if parsed is not None:
                match = re.search(r'(?:^|\n)Region=([^;\n]+)', metadata)
                if not match:
                    raise ModelProtocolError('R3 Question description lacks a trusted Region ID')
                self.tissue_status[match.group(1)] = parsed['tissue_status']
                if not hasattr(self, 'tissue_status_source'):
                    self.tissue_status_source = {}
                self.tissue_status_source[match.group(1)] = 'model_explicit'
                self.records[-1]['tissue_status'] = parsed['tissue_status']
                self.records[-1]['tissue_status_source'] = 'model_explicit'
                self.records[-1]['parsed_response'] = deepcopy(parsed)
                return parsed['findings']
            if attempt == 0:
                messages.append({'role': 'user', 'content':
                    self.rule_templates['QUESTION_FORMAT_RETRY_USER'].format(
                        previous_response=answer_content(output))})
        # A nonempty, completed observation without a parseable status remains
        # usable uncertain evidence. Never guess present/absent from pathology
        # prose, and never manufacture observations from incomplete JSON.
        observation = answer_content(output).strip()
        wrapper = re.fullmatch(r'<answer>\s*(.*?)\s*</answer>', observation, flags=re.S | re.I)
        if wrapper:
            observation = wrapper.group(1).strip()
        if (observation and not self.records[-1]['at_token_cap']
                and not re.match(r'^(?:\{|\[|```|<)', observation)):
            match = re.search(r'(?:^|\n)Region=([^;\n]+)', metadata)
            if not match:
                raise ModelProtocolError('R3 Question description lacks a trusted Region ID')
            if not hasattr(self, 'tissue_status_source'):
                self.tissue_status_source = {}
            self.tissue_status[match.group(1)] = 'uncertain'
            self.tissue_status_source[match.group(1)] = 'protocol-format-unknown'
            self.records[-1].update({
                'tissue_status': 'uncertain', 'tissue_status_source': 'protocol-format-unknown',
                'parsed_response': {'tissue_status': 'uncertain', 'findings': observation},
                'format_fallback': {'method': 'retain_completed_observation_with_unknown_status',
                                    'failed_format_attempts': 2, 'status_inferred': False},
            })
            return observation
        raise ModelProtocolError('Invalid tissue-status Question description after two attempts')

    def _registry(self, findings, review_context=None):
        context = review_context or {}
        quality = context.get('quality', {})
        statuses = dict(getattr(self, 'tissue_status', {}))
        statuses.update(context.get('tissue_status', {}))
        registry = {}
        for finding in findings:
            region = Region(**finding['region'])
            if region.region_id in registry and registry[region.region_id]['region'] != region.to_dict():
                raise ModelProtocolError('Conflicting metadata for Region ' + region.region_id)
            status = statuses.get(region.region_id, finding.get('tissue_status', 'not_assessed'))
            q = quality.get(region.region_id, {})
            # Only R3 enables quality/tissue exclusion; other arms retain the
            # original candidate policy and do not silently inherit filtering.
            usable = not ('R3' in self.rule_flags and
                          (status == 'absent' or q.get('hard_reject', False)))
            registry[region.region_id] = {
                'region': region.to_dict(), 'tissue_status': status,
                'quality': q if 'R3' in self.rule_flags else {}, 'usable': usable,
            }
            if 'R3' in self.rule_flags:
                registry[region.region_id]['tissue_status_source'] = getattr(
                    self, 'tissue_status_source', {}).get(region.region_id,
                    finding.get('tissue_status_source', 'not_assessed'))
        return registry

    def _narrative_chunks(self, findings, registry):
        chunks = []
        for finding in findings:
            region_id = finding['region']['region_id']
            if not registry[region_id]['usable']:
                chunks.append('Region ID: ' + region_id + '\nNo usable tissue evidence from this region.')
                continue
            generic = finding.get('generic_description')
            question = finding.get('question_description')
            if generic is not None or question is not None:
                parts = ['Region ID: ' + region_id]
                if generic:
                    parts.append('Generic observations:\n' + generic)
                if question:
                    parts.append('Question observations:\n' + question)
                chunks.append('\n'.join(parts))
            else:
                chunks.append('Region ID: ' + region_id + '\nObservations:\n' + finding['description'])
        return chunks

    def evidence_text(self, findings, question, extra='', requests=None, review_context=None):
        """Summarize narrative only, never summarize authoritative metadata."""
        registry = self._registry(findings, review_context)
        trusted = [{
            'region_id': name, 'bounds': [entry['region'][key] for key in ('x', 'y', 'width', 'height')],
            'scale_kind': entry['region']['scale_kind'], 'scale': entry['region']['scale'],
            'scale_source': entry['region']['magnification_source'],
            'parent_id': entry['region']['parent_id'], 'tissue_status': entry['tissue_status'],
            'quality': entry['quality'], 'usable': entry['usable'],
        } for name, entry in registry.items()]
        if 'R3' in self.rule_flags:
            for item in trusted:
                source = registry[item['region_id']]['tissue_status_source']
                if source == 'protocol-format-unknown':
                    item['tissue_status_source'] = source
        prefix = 'Trusted Region metadata (authoritative for scale, bounds and validity):\n' + _dump(trusted)
        chunks = self._narrative_chunks(findings, registry)
        algorithm = self.config['algorithm']
        budget = self.context_limit - self.generation['final_tokens'] - len(self.tokenizer.encode(extra + question)) - 2048
        if budget < 1024:
            raise ModelProtocolError('Reasoning history alone exceeds context')
        for level in range(5):
            narrative = '\n\n'.join(chunks)
            text = prefix + '\n\nObserved descriptions:\n' + narrative
            fits = len(self.tokenizer.encode(text)) <= budget
            if requests is not None:
                fits = all(len(self.tokenizer.encode(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))) + limit <= self.context_limit
                    for messages, limit in requests(text))
            if fits and (level > 0 or len(chunks) <= algorithm['summary_threshold']):
                return text
            previous = len(self.tokenizer.encode(narrative))
            summarized = []
            for start in range(0, len(chunks), algorithm['summary_chunk_size']):
                group = chunks[start:start + algorithm['summary_chunk_size']]
                summary = self._text([
                    {'role': 'system', 'content': self.rule_templates['SUMMARY_SYSTEM']},
                    {'role': 'user', 'content': 'Question: ' + question + '\n' + '\n\n'.join(group)}],
                    self.generation['summary_tokens'], 'summary', greedy=True)
                summarized.append(summary)
            chunks = summarized
            if len(self.tokenizer.encode('\n\n'.join(chunks))) >= previous:
                raise ModelProtocolError('Summary did not reduce overlong evidence')
        raise ModelProtocolError('Unable to fit evidence without dropping trusted metadata')

    def _retry_user(self, stage, context, previous_response):
        logical_stage = 'missing' if stage == 'missing_info' else stage
        names = {'predict': 'PREDICT_FORMAT_RETRY_USER', 'reflect': 'REFLECT_FORMAT_RETRY_USER',
                 'missing': 'MISSING_FORMAT_RETRY_USER', 'final': 'FINAL_FORMAT_RETRY_USER'}
        return self.rule_templates['JSON_FORMAT_RETRY_USER'].format(
            schema=_dump(schemas_for_flags(self.rule_flags)[logical_stage]), context=_dump(context or {}),
            stage_instruction=self.rule_templates[names[logical_stage]], previous_response=previous_response)

    def _json(self, system, prompt, tokens, stage, keys, greedy=False, validator=None, retry_context=None):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
        for attempt in range(self.generation['retries'] + 1):
            raw = self._text(messages, tokens, stage, greedy)
            parsed = parse_json_object(raw)
            if stage == 'missing_info' and parsed is not None:
                original = deepcopy(parsed)
                parsed, normalized, unused_target = self._normalize_missing_fields(parsed)
                self.records[-1]['parsed_raw_response'] = original
                if normalized:
                    self.records[-1]['value_normalization'] = normalized
                if unused_target is not None:
                    self.records[-1]['unused_target'] = unused_target
            elif stage == 'predict' and attempt > 0 and parsed is not None and 'thinking_steps' not in parsed:
                # A retry may omit this reporting field while retaining its
                # clinical answer and rule fields. Do not invent reasoning.
                self.records[-1]['parsed_raw_response'] = deepcopy(parsed)
                parsed['thinking_steps'] = ''
                self.records[-1]['value_normalization'] = {
                    'thinking_steps': {'from': '<omitted>', 'to': '',
                                       'reason': 'reporting_field_omitted_on_retry'}}
            valid = parsed is not None and all(key in parsed for key in keys)
            if valid and (validator is None or validator(parsed)):
                return parsed
            messages = deepcopy(messages[:2])
            messages.append({'role': 'user', 'content': self._retry_user(stage, retry_context, raw)})
        raise ModelProtocolError(f'Invalid {stage} response after {attempt + 1} attempts')

    def _normalize_missing_fields(self, value):
        """Normalize optional/unused action fields without changing the action."""
        normalized, changes, unused_target = deepcopy(value), {}, None
        no_zoom = isinstance(value.get('zoom_recommendation'), str) and value['zoom_recommendation'].lower() == 'no'
        if no_zoom:
            level = value.get('recommended_zoom_level', '<omitted>')
            if level is not None:
                changes['recommended_zoom_level'] = {'from': level, 'to': None, 'reason': 'unused_for_zoom_no'}
            normalized['recommended_zoom_level'] = None
        if 'R5' in self.rule_flags:
            target = value.get('target_region', '<omitted>')
            if no_zoom:
                if target is not None:
                    changes['target_region'] = {'from': target, 'to': None, 'reason': 'unused_for_zoom_no'}
                if 'target_region' in value and target not in (None, '', 'null'):
                    unused_target = deepcopy(target)
                normalized['target_region'] = None
            elif 'target_region' not in value or target == 'null':
                changes['target_region'] = {'from': target, 'to': None}
                normalized['target_region'] = None
        return normalized, changes, unused_target

    def _evidence_for_v2(self, findings, system, render, tokens, question, context,
                         retry_stage=None, retry_context=None):
        retry = (self._retry_user(retry_stage, retry_context, '') if retry_stage is not None
                 else self.rule_templates['RETRY_USER'])
        def requests(description):
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': render(description)}]
            # A retry contains the previous capped output plus a new capped
            # generation. Reserve both to avoid a retry-only context overflow.
            limit = 2 * tokens + 32 if retry_stage is not None else tokens
            return [(messages + [{'role': 'user', 'content': retry}], limit)]
        return self.evidence_text(findings, question, requests=requests, review_context=context)

    def _valid_predict(self, value, registry, choices=None):
        if not _nonempty(value.get('answer')) or not isinstance(value.get('thinking_steps'), str):
            return False
        if 'R1' in self.rule_flags:
            features = value.get('required_features')
            if not isinstance(features, list) or not features:
                return False
            names = set()
            for feature in features:
                if (not isinstance(feature, dict) or not _nonempty(feature.get('feature'))
                        or feature.get('evidence_type') not in REQUIRED_SCALES
                        or feature['feature'] in names):
                    return False
                names.add(feature['feature'])
        if 'R2' in self.rule_flags:
            if not _nonempty(value.get('alternative_answer')) or not _nonempty(value.get('discriminating_feature')):
                return False
            conflicts = value.get('conflicts')
            if not isinstance(conflicts, list):
                return False
            for conflict in conflicts:
                if (not isinstance(conflict, dict) or not _nonempty(conflict.get('region_id'))
                        or not _nonempty(conflict.get('feature')) or not isinstance(conflict.get('claims'), list)
                        or len(conflict['claims']) < 2 or not all(_nonempty(c) for c in conflict['claims'])):
                    return False
        return True

    def _valid_reflect(self, value):
        if str(value.get('sufficient')).lower() not in {'yes', 'no'}:
            return False
        if 'R1' in self.rule_flags:
            checks = value.get('checks')
            if not isinstance(checks, list):
                return False
            for check in checks:
                scalar = check.get('region_id') if isinstance(check, dict) else None
                refs = check.get('region_ids') if isinstance(check, dict) else None
                valid_refs = ((_nonempty(scalar) and (refs is None or refs == [scalar]))
                              or (scalar is None and isinstance(refs, list) and all(_nonempty(r) for r in refs)))
                if (not isinstance(check, dict) or not _nonempty(check.get('feature'))
                        or check.get('evidence_type') not in REQUIRED_SCALES
                        or not valid_refs
                        or type(check.get('assessable')) is not bool
                        or type(check.get('coverage_sufficient')) is not bool):
                    return False
        if 'R2' in self.rule_flags:
            conflicts = value.get('unresolved_conflicts')
            if not isinstance(conflicts, list):
                return False
            if any(not isinstance(c, dict) or not _nonempty(c.get('region_id'))
                   or not _nonempty(c.get('feature')) for c in conflicts):
                return False
        if 'R4' in self.rule_flags and value.get('target_observed') not in {'Yes', 'No', 'Unknown'}:
            return False
        return True

    def _normalize_reflection(self, reflection):
        normalized = deepcopy(reflection)
        if 'R1' in self.rule_flags:
            for check in normalized['checks']:
                if 'region_id' in check:
                    check['region_ids'] = [check.pop('region_id')]
        return normalized

    def _apply_type_policy(self, prediction):
        normalized, adjustments = deepcopy(prediction), []
        if 'R1' in self.rule_flags:
            for required in normalized['required_features']:
                name = re.sub(r'[_\s]+', ' ', required['feature']).lower()
                explicit_mitosis = re.search(r'\b(?:mitosis|mitoses|mitotic\s+(?:activity|counts?|figures?))\b', name)
                if explicit_mitosis and required['evidence_type'] != 'fine_detail':
                    adjustments.append({'feature': required['feature'], 'from': required['evidence_type'],
                                        'to': 'fine_detail', 'required_scale': REQUIRED_SCALES['fine_detail']})
                    required['evidence_type'] = 'fine_detail'
        return normalized, adjustments

    def _effective_reflection(self, prediction, reflection, registry, choices=None):
        failures, scale_failures = [], []
        if 'R1' in self.rule_flags:
            checks = reflection['checks']
            for required in prediction['required_features']:
                feature, kind = required['feature'], required['evidence_type']
                same_feature = [c for c in checks if c['feature'] == feature]
                matching = [c for c in same_feature if c['evidence_type'] == kind]
                floor = REQUIRED_SCALES[kind]
                single_check = len(same_feature) == 1 and len(matching) == 1
                if len(same_feature) != 1:
                    failures.append({'feature': feature, 'reason': 'missing_or_duplicate_check'})
                elif len(matching) != 1:
                    failures.append({'feature': feature, 'reason': 'mismatched_check_type'})
                refs = list(dict.fromkeys(name for check in same_feature for name in check['region_ids']))
                if not refs or any(name not in registry for name in refs):
                    failures.append({'feature': feature, 'reason': 'missing_or_invalid_region'})
                usable_refs = [name for name in refs if name in registry and registry[name]['usable']]
                if not usable_refs:
                    failures.append({'feature': feature, 'reason': 'no_usable_region'})
                    continue
                if single_check:
                    check = matching[0]
                    if not check['assessable']:
                        failures.append({'feature': feature, 'reason': 'unassessable'})
                    if not check['coverage_sufficient']:
                        failures.append({'feature': feature, 'reason': 'insufficient_field_coverage'})
                adequate = [name for name in usable_refs if
                            registry[name]['region']['scale_kind'] == 'physical' and
                            registry[name]['region']['scale'] >= floor]
                if not adequate:
                    failures.append({'feature': feature, 'reason': 'scale_insufficient'})
                    scale_failures.append({'feature': feature, 'evidence_type': kind,
                                           'region_ids': usable_refs, 'required_scale': floor})
        if 'R2' in self.rule_flags:
            if choices:
                answer = answer_label(prediction['answer'], choices)
                alternative = answer_label(prediction['alternative_answer'], choices)
                if answer is None:
                    failures.append({'reason': 'invalid_primary_answer',
                                     'primary_answer': prediction['answer']})
                if alternative is None or answer == alternative:
                    failures.append({'reason': 'invalid_differential', 'primary_answer': prediction['answer'],
                                     'alternative_answer': prediction['alternative_answer']})
            for conflict in prediction['conflicts']:
                if conflict['region_id'] not in registry:
                    failures.append({'reason': 'invalid_conflict_source',
                                     'region_id': conflict['region_id'], 'feature': conflict['feature'],
                                     'claims': deepcopy(conflict['claims'])})
            for conflict in reflection['unresolved_conflicts']:
                failures.append({'feature': conflict['feature'], 'region_id': conflict['region_id'],
                                 'reason': 'unresolved_conflict' if conflict['region_id'] in registry
                                 else 'invalid_conflict_region'})
        no_usable = 'R3' in self.rule_flags and not any(r['usable'] for r in registry.values())
        if no_usable:
            failures.append({'reason': 'no_usable_tissue'})
        effective = 'No' if failures or reflection['sufficient'].lower() == 'no' else 'Yes'
        return effective, failures, scale_failures, no_usable

    def _context_blocks(self, context):
        blocks = []
        if 'R1' in self.rule_flags:
            blocks.append('Supplied minimum magnification policy:\n' + _dump(REQUIRED_SCALES))
        if 'R4' in self.rule_flags:
            new_regions = []
            for finding in context.get('last_new_findings', []):
                if not isinstance(finding, dict):
                    raise ModelProtocolError('Retrieval-progress finding must have a Region reference')
                region = finding.get('region', {})
                region_id = region.get('region_id') if isinstance(region, dict) else region
                if not region_id:
                    region_id = finding.get('region_id')
                if not _nonempty(region_id):
                    raise ModelProtocolError('Retrieval-progress finding lacks actual Region ID')
                new_regions.append({'region_id': region_id})
            blocks.append('Retrieval progress from actual executed actions:\n' + _dump({
                'last_request': context.get('last_request'),
                'last_new_findings': new_regions,
                'no_progress_count': context.get('no_progress_count', 0),
            }) + '\nAssess new evidence at these IDs in the supplied descriptions.')
        return '\n\n' + '\n\n'.join(blocks) if blocks else ''

    def evaluate(self, findings, sample, scale_kind, allowed_scales, review_context=None):
        context = review_context or {}
        registry = self._registry(findings, context)
        templates = self.rule_templates
        fields = {'question': sample['question'], 'choices': choices_text(sample['choices'])}
        retry_context = {'choices': sample['choices']}
        context_blocks = self._context_blocks(context)
        render = lambda description: templates['PREDICT_USER'].format(description=description, **fields) + context_blocks
        description = self._evidence_for_v2(findings, templates['PREDICT_SYSTEM'], render,
                                             self.generation['step_a_tokens'], sample['question'], context,
                                             retry_stage='predict', retry_context=retry_context)
        prediction = self._json(templates['PREDICT_SYSTEM'], render(description), self.generation['step_a_tokens'],
                                'predict', predict_keys(self.rule_flags),
                                validator=lambda value: self._valid_predict(value, registry, sample['choices']), retry_context=retry_context)
        prediction_raw = deepcopy(prediction)
        prediction, type_adjustments = self._apply_type_policy(prediction)
        render = lambda description: templates['REFLECT_USER'].format(
            description=description, answer=_dump(prediction), **fields) + context_blocks
        description = self._evidence_for_v2(findings, templates['REFLECT_SYSTEM'], render,
                                             self.generation['step_b_tokens'], sample['question'], context,
                                             retry_stage='reflect', retry_context={**retry_context, 'required_features': prediction.get('required_features', [])})
        reflection_start, reflection_failure = len(self.records), None
        try:
            reflection = self._json(templates['REFLECT_SYSTEM'], render(description), self.generation['step_b_tokens'],
                                    'reflect', reflect_keys(self.rule_flags), validator=self._valid_reflect,
                                    retry_context={**retry_context, 'required_features': prediction.get('required_features', [])})
            reflection_raw = deepcopy(reflection)
        except ModelProtocolError as error:
            attempts = self.records[reflection_start:]
            if (not self.rule_flags.intersection({'R1', 'R2'})
                    or str(error) != 'Invalid reflect response after 2 attempts'
                    or len(attempts) != 2 or any(record.get('stage') != 'reflect' for record in attempts)):
                raise
            # Unknown format cannot establish sufficiency. Retain the original
            # conflict sources; never repair a truncated prefix into model JSON.
            reflection = {'sufficient': 'No'}
            if 'R2' in self.rule_flags:
                reflection['unresolved_conflicts'] = [
                    {'region_id': conflict['region_id'], 'feature': conflict['feature']}
                    for conflict in prediction['conflicts']]
            if 'R1' in self.rule_flags:
                reflection['checks'] = []
            if 'R4' in self.rule_flags:
                reflection['target_observed'] = 'Unknown'
            reflection_raw = None
            reflection_failure = {'reason': str(error), 'attempts': [
                {'record_index': reflection_start + index, 'raw_response': record.get('raw_response'),
                 'at_token_cap': record.get('at_token_cap')}
                for index, record in enumerate(attempts)]}
        reflection = self._normalize_reflection(reflection)
        effective, failures, scale_failures, no_usable = self._effective_reflection(prediction, reflection, registry, sample['choices'])
        decision = {**prediction, **reflection, 'reflection_raw': reflection_raw,
                    'sufficient': effective, 'effective_sufficient': effective,
                    'evidence_failures': failures, 'scale_failures': scale_failures}
        if reflection_failure is not None:
            decision['reflection_source'] = 'protocol-format-unknown'
            decision['reflection_protocol_failure'] = reflection_failure
        if 'R1' in self.rule_flags:
            decision['prediction_raw'] = prediction_raw
            decision['type_policy_adjustments'] = type_adjustments
        if effective == 'Yes':
            return decision
        parents = context.get('eligible_zoom_parents', [])
        eligible = {p['region_id']: p for p in parents}
        scale_description = ('absolute magnification' if scale_kind == 'physical' else
                             'relative crop factor; physical magnification is unknown')
        effective_reflection = {**reflection, 'effective_sufficient': effective,
                                'evidence_failures': failures, 'scale_failures': scale_failures}
        if reflection_failure is not None:
            effective_reflection.update(reflection_source=decision['reflection_source'],
                                        reflection_protocol_failure=reflection_failure)
        missing_context = dict(context)
        if 'R4' in self.rule_flags and context.get('last_request') is not None:
            missing_context['no_progress_count'] = (0 if reflection['target_observed'] == 'Yes'
                                                    else int(context.get('no_progress_count', 0)) + 1)
            decision['current_no_progress_count'] = missing_context['no_progress_count']
        missing_blocks = self._context_blocks(missing_context)
        def render(description):
            prompt = templates['MISSING_USER'].format(
                description=description, prediction=_dump(prediction), reflection=_dump(effective_reflection),
                scale_description=scale_description, allowed_scales=allowed_scales, **fields) + missing_blocks
            if 'R5' in self.rule_flags:
                prompt += '\n\nEligible zoom parents:\n' + _dump(parents)
            return prompt
        description = self._evidence_for_v2(findings, templates['MISSING_SYSTEM'], render,
                                             self.generation['step_c_tokens'], sample['question'], context,
                                             retry_stage='missing_info', retry_context={**retry_context,
                                                 'allowed_scales': list(allowed_scales), 'eligible_parent_ids': list(eligible)})
        def valid_missing(value):
            if (not _nonempty(value.get('missing_info')) or not isinstance(value.get('zoom_reason'), str)
                    or str(value.get('zoom_recommendation')).lower() not in {'yes', 'no'}):
                return False
            level = value.get('recommended_zoom_level')
            if isinstance(level, str) and level in {str(v) for v in allowed_scales}:
                level = value['recommended_zoom_level'] = int(level)
            if level == 'None':
                level = value['recommended_zoom_level'] = None
            zoom_yes = value['zoom_recommendation'].lower() == 'yes'
            if level is not None and (type(level) is not int or level not in allowed_scales):
                return False
            if zoom_yes and level not in allowed_scales:
                return False
            if 'R4' in self.rule_flags and value.get('evidence_type') not in REQUIRED_SCALES:
                return False
            if 'R5' in self.rule_flags:
                target = value.get('target_region')
                if target is not None and (not isinstance(target, str) or target not in eligible or not zoom_yes):
                    return False
            return True
        missing = self._json(templates['MISSING_SYSTEM'], render(description), self.generation['step_c_tokens'],
                             'missing_info', missing_keys(self.rule_flags), validator=valid_missing,
                             retry_context={**retry_context, 'allowed_scales': list(allowed_scales),
                                            'eligible_parent_ids': list(eligible)})
        decision['missing_raw'] = deepcopy(self.records[-1].get('parsed_raw_response', missing))
        if self.records[-1].get('value_normalization'):
            decision['missing_value_normalization'] = deepcopy(self.records[-1]['value_normalization'])
        if self.records[-1].get('unused_target') is not None:
            decision['unused_target'] = deepcopy(self.records[-1]['unused_target'])
        target_parent = missing.get('target_region') if 'R5' in self.rule_flags else None
        execution_parents = ({target_parent} if target_parent is not None else
                             {p['region_id'] for p in context.get('current_zoom_parents', [])})
        floor_candidates = []
        unavailable_targets = []
        for failure in scale_failures:
            relevant_parents = []
            cited_parents = []
            for name in failure['region_ids']:
                region = registry[name]['region']
                parent_id = region.get('parent_id') or name
                cited_parents.append(parent_id)
                # The cited child is already required to be usable. A low-power
                # tissue-status error must not forbid rechecking its parent.
                if parent_id in eligible and parent_id in execution_parents:
                    relevant_parents.append(parent_id)
            if relevant_parents:
                floor_candidates.append(failure['required_scale'])
            else:
                unavailable_targets.append({
                    'feature': failure['feature'], 'required_scale': failure['required_scale'],
                    'cited_parent_regions': sorted(set(cited_parents)),
                    'execution_parent_regions': sorted(execution_parents),
                    'reason': 'NeedsHigherScaleButNoCurrentTarget',
                })
        if unavailable_targets:
            decision['scale_target_unavailable'] = unavailable_targets
        if no_usable:
            missing.update(zoom_recommendation='No', recommended_zoom_level=None)
            if 'R5' in self.rule_flags:
                missing['target_region'] = None
            decision['action_override'] = 'No usable tissue: Explore rather than Zoom.'
        elif floor_candidates:
            floor = max(floor_candidates)
            allowed = sorted(v for v in allowed_scales if v >= floor)
            if allowed:
                requested = missing['recommended_zoom_level']
                level = max(allowed[0], requested or allowed[0])
                missing.update(zoom_recommendation='Yes', recommended_zoom_level=level)
                decision['action_override'] = 'Scale-insufficient relevant evidence: enforce minimum Zoom scale.'
        missing['zoom_level'] = missing['recommended_zoom_level']
        return {**decision, **missing}

    def conclude(self, findings, trajectory, sample, review_context=None):
        context = review_context or {}
        templates = self.rule_templates
        history = _dump([{k: event[k] for k in ('iteration', 'action', 'decision')} for event in trajectory])
        render = lambda description: templates['FINAL_USER'].format(
            question=sample['question'], choices=choices_text(sample['choices']),
            description=description, history=history)
        description = self._evidence_for_v2(findings, templates['FINAL_SYSTEM'], render,
                                             self.generation['final_tokens'], sample['question'], context,
                                             retry_stage='final', retry_context={'choices': sample['choices']})
        return self._json(templates['FINAL_SYSTEM'], render(description), self.generation['final_tokens'],
                          'final', ['answer', 'explanation'], greedy=True,
                          validator=lambda value: _nonempty(value.get('answer'))
                          and isinstance(value.get('explanation'), str)
                          and (not sample['choices'] or answer_label(value['answer'], sample['choices']) is not None),
                          retry_context={'choices': sample['choices']})
