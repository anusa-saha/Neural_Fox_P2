MODELS = {
    "gemma2_9b_it": {
        "hf_id": "google/gemma-2-9b-it",
        "family": "gemma",
        "n_layers": 42,
        "hidden_size": 3584,
        "hook_module_path": "model.layers.{layer}",
        "gemma_variant": "pt",          
        "gemma_width": "16k",
    },
    "llama3_1_8b_instruct": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "n_layers": 32,
        "hidden_size": 4096,
        "hook_module_path": "model.layers.{layer}",
        "llama_sae_release": "llama_scope_lxr_8x",
        "llama_sae_id_template": "l{layer}r_8x",
    },
    "qwen3_8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "family": "qwen",
        "n_layers": 36,
        "hidden_size": 4096,
        "hook_module_path": "model.layers.{layer}",
        "sae_repo": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
        "sae_filename_template": "layer{layer}.sae.pt",
        "sae_layer_index_base": 0,
    },
    "qwen3_5_9b": {
        "hf_id": "Qwen/Qwen3.5-9B",  
        "family": "qwen",
        "n_layers": 40,
        "hidden_size": 4096,
        "hook_module_path": "model.layers.{layer}",
        "sae_repo": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100",
        "sae_filename_template": "layer{layer}.sae.pt",
        "sae_layer_index_base": 0,
    },
}

# Target languages from the paper -> (FLORES+ config code, ISO-639-1 code for LID)
LANGUAGES = {
    "hi": {"flores": "hin_Deva", "lid": "hi"},
    "es": {"flores": "spa_Latn", "lid": "es"},
    "zh": {"flores": "cmn_Hans", "lid": "zh"},
    "bn": {"flores": "ben_Beng", "lid": "bn"},
    "te": {"flores": "tel_Telu", "lid": "te"},
}
ENGLISH_FLORES = "eng_Latn"
ENGLISH_LID = "en"
FLORES_SPLITS = ("dev", "devtest")

