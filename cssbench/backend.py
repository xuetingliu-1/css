
import copy
import os
from .models import profile_for


def prepare_messages(messages, family):
    result = copy.deepcopy(messages)
    if not result or any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str) for m in result):
        raise ValueError('Expected nonempty text-only chat messages')
    if family == 'llama_user_only' and result[0]['role'] == 'system':
        if len(result) != 2 or result[1]['role'] != 'user':
            raise ValueError('UltraMedical supports a single user turn in this experiment')
        result = [{'role': 'user', 'content': result[0]['content'] + '\n\n' + result[1]['content']}]
    # Prevent Qwen2.5 from silently inserting its own assistant persona.
    if family == 'qwen25' and result[0]['role'] != 'system':
        result.insert(0, {'role': 'system', 'content': ''})
    return result


def task_only_template(template, family):
    if not template:
        raise ValueError('The local tokenizer has no chat template; use a complete model checkpoint')
    if family == 'llama_metadata':
        # Keep native Llama role delimiters, but remove metadata that would change
        # experiment content. Do not replace the checkpoint's entire template.
        for line in ['{{- "Cutting Knowledge Date: December 2023\\n" }}', '{{- "Today Date: " + date_string + "\\n\\n" }}']:
            template = template.replace(line, '')
        if 'Cutting Knowledge Date:' in template or '{{- "Today Date:' in template:
            raise ValueError('Unrecognized Llama metadata template; audit before running')
    return template


class ChatTokenizer:
    def __init__(self, raw, key, family):
        self.raw, self.key, self.family = raw, key, family
        self.eos_token_id = raw.eos_token_id
        self.pad_token_id = raw.pad_token_id if raw.pad_token_id is not None else raw.eos_token_id
        if self.pad_token_id is None:
            raise ValueError('Tokenizer must define EOS or PAD')
        self.padding_side = 'left'
        self.stop_token_ids = [self.eos_token_id] if self.eos_token_id is not None else []
        end = '<|eot_id|>' if family.startswith('llama') else '<end_of_turn>' if family == 'gemma3' else None
        if end:
            vocab = raw.get_vocab()
            if end not in vocab:
                raise ValueError(f'Missing required turn terminator {end}')
            self.stop_token_ids.append(vocab[end])
        self.stop_token_ids = sorted(set(self.stop_token_ids))
        self.template = None if family == 'mistral3' else task_only_template(raw.chat_template, family)

    def __getattr__(self, key):
        return getattr(self.raw, key)

    def _apply_one(self, messages, tokenize):
        messages = prepare_messages(messages, self.family)
        if self.family == 'mistral3':
            # MistralCommonBackend rejects add_generation_prompt, and already
            # opens the assistant generation boundary itself.
            result = self.raw.apply_chat_template(messages, tokenize=tokenize)
            if tokenize and hasattr(result, 'keys'):
                result = result['input_ids']
            if tokenize and result and isinstance(result[0], list):
                if len(result) != 1:
                    raise ValueError('Unexpected multi-row tokenizer output')
                result = result[0]
            return result
        kwargs = {'chat_template': self.template, 'add_generation_prompt': True, 'tokenize': tokenize}
        if self.family == 'qwen3':
            kwargs['enable_thinking'] = False
        return self.raw.apply_chat_template(messages, **kwargs)

    def render(self, messages):
        return self._apply_one(messages, tokenize=False)

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=True,
                            return_tensors=None, return_dict=False, padding=False, **kwargs):
        if kwargs:
            raise ValueError(f'Unsupported template options: {sorted(kwargs)}')
        if not add_generation_prompt:
            raise ValueError('Inference adapter always opens an assistant generation turn')
        batched = bool(conversation and isinstance(conversation[0], (list, tuple)))
        conversations = conversation if batched else [conversation]
        if not tokenize:
            rendered = [self.render(m) for m in conversations]
            return rendered if batched else rendered[0]
        rows = [self._apply_one(m, tokenize=True) for m in conversations]
        if return_tensors is None:
            result = rows if batched else rows[0]
            return {'input_ids': result} if return_dict else result
        if return_tensors != 'pt':
            raise ValueError('Only PyTorch tensors are supported')
        import torch
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row):] = torch.tensor(row, dtype=torch.long)
            mask[i, -len(row):] = 1
        return {'input_ids': ids, 'attention_mask': mask} if return_dict else ids


def load_tokenizer(model_path, key=None):
    key, family = profile_for(model_path, key)
    if family == 'mistral3':
        try:
            from transformers import MistralCommonBackend
        except ImportError as e:
            raise ImportError('Ministral requires Transformers with MistralCommonBackend and mistral-common installed') from e
        raw = MistralCommonBackend.from_pretrained(model_path)
    else:
        from transformers import AutoTokenizer
        raw = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    return ChatTokenizer(raw, key, family)


def encode_batch(tokenizer, messages_list):
    return tokenizer.apply_chat_template(messages_list, return_tensors='pt', return_dict=True, padding=True)


def load_model(model_path, dtype=None, device_map='auto', load_in_4bit=False,
               dequantize_fp8_to_bf16=False):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    _, family = profile_for(model_path)
    if family == 'mistral3':
        from transformers import Mistral3ForConditionalGeneration
        cls = Mistral3ForConditionalGeneration
    elif family == 'gemma3':
        from transformers import Gemma3ForConditionalGeneration
        cls = Gemma3ForConditionalGeneration
    else:
        cls = AutoModelForCausalLM
    if load_in_4bit and dequantize_fp8_to_bf16:
        raise ValueError('Choose one quantization option, not both')
    if dequantize_fp8_to_bf16 and family != 'mistral3':
        raise ValueError('FP8 dequantization is only configured for the Ministral checkpoint')
    kwargs = {'device_map': device_map}
    if dtype is not None:
        kwargs['torch_dtype'] = dtype
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype or torch.float16)
    if dequantize_fp8_to_bf16:
        from transformers import FineGrainedFP8Config
        kwargs['quantization_config'] = FineGrainedFP8Config(dequantize=True)
    model = cls.from_pretrained(model_path, **kwargs).eval()

    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.0
    return model


def load_css_model(model_path, gpu_id, dtype, dequantize_fp8_to_bf16):
    import torch
    # Respect Slurm's device mapping; never overwrite an existing allocation.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    types = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    tok = load_tokenizer(model_path)
    return load_model(model_path, dtype=types[dtype], dequantize_fp8_to_bf16=dequantize_fp8_to_bf16), tok
