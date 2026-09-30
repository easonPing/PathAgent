"""Published wording, with explicitly attributed unpublished message wrappers."""
import hashlib

PROTOCOL = 'paper-prompts-nonthinking-2x-v1'
GENERIC_PROMPT = 'Please describe the pathology features in this image.'
QUESTION_PROMPT = 'Please describe the pathology features related to the question: {question} in this image.'
PERCEPTOR_SYSTEM = ('A conversation between a curious user and an AI medical assistant specialized in pathology image analysis. '
    'The assistant can interpret pathology images, describe observed features, and provide possible explanations based on medical knowledge, '
    'but will never give a definitive diagnosis or prescribe treatment. The assistant must always maintain a polite, clear, and professional tone. '
    'All answers should be supported by established, reliable medical sources. The assistant should carefully consider visual details in pathology images, '
    'such as cell morphology, staining patterns, and tissue architecture.')
PREDICT_SYSTEM = ('You are an expert AI pathology assistant. Your task is trying to answer the question step-by-step based on the patch descriptions. '
    'Output ONLY a JSON object: {"answer": "the final predicted answer (string)", "thinking_steps": "your detailed reasoning, step-by-step (string)"}.')
REFLECT_SYSTEM = ('You are an expert AI pathology assistant. Your task is to judge whether the current patch descriptions are sufficient to confidently support the answer. '
    'Output ONLY a JSON object: {"sufficient": "Yes" or "No"}.')
MISSING_SYSTEM = ('You are an expert AI pathology assistant. Your task is to specify what visual evidence is missing and whether zooming in could help obtain that evidence. '
    'Output ONLY a JSON object: {"missing_info": "noun phrase", "zoom_recommendation": "Yes" or "No", '
    '"recommended_zoom_level": "None" or an integer like 10 or 20 or 40, "zoom_reason": "brief reason why zooming helps"}.')
FINAL_SYSTEM = ('You are an expert slide-level pathology assistant. You will be given a question and detailed patch-level descriptions of a pathology slide. '
    'Your task is to infer the specific slide-level diagnostic result based on the provided evidence — not to define or explain the medical term itself. '
    'The answer should directly reflect the information observable in the slide, such as biomarker expression level, presence or absence of features, or a numeric measurement.')
SUMMARY_SYSTEM = "Summarize visible pathological findings, preserving region coordinates and each region's scale. Do not give a final answer or diagnosis."
PREDICT_USER = "--- Patch Descriptions ---\n{description}\n--- End ---\nQuestion: {question}{choices}\nNow output the JSON with 'answer' and 'thinking_steps'."
REFLECT_USER = 'Descriptions:\n{description}\n\nQuestion: {question}{choices}\n\nPrevious answer and reasoning:\n{answer}\n\nOnly return a JSON object: {{"sufficient": "Yes" or "No"}}'
MISSING_USER = 'Descriptions:\n{description}\n\nQuestion: {question}{choices}\n\nNow provide the JSON for missing info and zoom recommendation.\nScale means {scale_description}. Available zoom levels: {allowed_scales}.'
FINAL_USER = ('Question: {question}{choices}\n\nNow, based on the following patch-level descriptions of the slide, determine the **slide-level result** that directly answers the question.\n\n'
    '--- Patch Descriptions ---\n{description}\n--- End of Descriptions ---\n\nPrior inferences and actions:\n{history}\n\n'
    'Answer in JSON format with keys "answer" and "explanation" (both strings). If choices are provided, answer must be exactly one option text or its label.')
RETRY_USER = 'Return one valid JSON object with the required keys and valid field values only.'


def prompt_manifest():
    templates = {}
    for name, value in globals().items():
        if name.endswith(('_SYSTEM', '_USER', '_PROMPT')) and isinstance(value, str):
            published = name in {'GENERIC_PROMPT', 'QUESTION_PROMPT', 'PERCEPTOR_SYSTEM', 'PREDICT_SYSTEM', 'REFLECT_SYSTEM', 'MISSING_SYSTEM', 'FINAL_SYSTEM'}
            templates[name] = {'text': value, 'sha256': hashlib.sha256(value.encode()).hexdigest(),
                'published': published, 'source': ('arxiv:2511.17052v1 sec/3_method.tex §3.1' if name.endswith('_PROMPT') else
                'arxiv:2511.17052v1 sec/X_suppl.tex §Prompt') if published else
                'author 8c2abee models/inference.py wrapper adapted to Algorithm 1; summary/retry are engineering defaults'}
    return {'protocol': PROTOCOL, 'templates': templates,
            'notes': ['Zoom reuses the published question template.', 'Executor user wrappers are not fully published.',
                      'ROI scale adapter is relative, not physical magnification.']}
