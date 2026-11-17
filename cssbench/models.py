
MODELS = {
    'mmed': ('Henrychur/MMed-Llama-3-8B-EnIns', 'llama'),
    'meerkat': ('dmis-lab/llama-3-meerkat-8b-v1.0', 'llama'),
    'ultramedical': ('TsinghuaC3I/Llama-3-8B-UltraMedical', 'llama_user_only'),
    'medreason': ('UCSC-VLAA/MedReason-8B', 'llama_metadata'),
    'medgemma': ('google/medgemma-1.5-4b-it', 'gemma3'),
    'llama31': ('meta-llama/Llama-3.1-8B-Instruct', 'llama_metadata'),
    'ministral': ('mistralai/Ministral-3-8B-Instruct-2512', 'mistral3'),
    'qwen25': ('Qwen/Qwen2.5-7B-Instruct', 'qwen25'),
    'qwen3': ('Qwen/Qwen3-8B', 'qwen3'),
}
_SELECTED = None

def select_model(key):
    global _SELECTED
    if key not in MODELS:
        raise ValueError(f'Unknown model profile: {key}')
    _SELECTED = key

def profile_for(path, key=None):
    if key is not None:
        return key, MODELS[key][1]
    matches = [k for k, (repo, _) in MODELS.items() if str(path).rstrip('/').split('/')[-1].lower() == repo.split('/')[-1].lower()]
    if matches:
        k = matches[0]
    elif _SELECTED:
        k = _SELECTED
    else:
        raise ValueError('Select a model profile with --model, or pass key= to load_tokenizer.')
    return k, MODELS[k][1]
