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
        "sae_layer_index_base": 1,
    },
    "qwen3_5_9b": {
        "hf_id": "Qwen/Qwen3.5-9B",  
        "family": "qwen",
        "n_layers": 40,
        "hidden_size": 4096,
        "hook_module_path": "model.layers.{layer}",
        "sae_repo": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100",
        "sae_filename_template": "layer{layer}.sae.pt",
        "sae_layer_index_base": 1,
    },
    "gemma3_12b_it": {
        "hf_id": "google/gemma-3-12b-it",
        "family": "gemma3",
        "n_layers": 48,
        "hidden_size": 3840,
        "hook_module_path": "model.layers.{layer}",
        "gemma3_scope_release": "gemma-scope-2-12b-it-resid_post",
        "gemma3_scope_width": "64k",   
        "gemma3_scope_l0": "medium",   
    },
    "gemma3_4b_it": {
        "hf_id": "google/gemma-3-4b-it",
        "family": "gemma3",
        "n_layers": 34,
        "hidden_size": 2560,
        "hook_module_path": "model.layers.{layer}",
        "gemma3_scope_release": "gemma-scope-2-4b-it-resid_post",
        "gemma3_scope_width": "64k",
        "gemma3_scope_l0": "medium",
    },
}


LANGUAGES = {
    "hi": {"flores": "hin_Deva", "lid": "hi"},
    "es": {"flores": "spa_Latn", "lid": "es"},
    "zh": {"flores": "zho_Hans", "lid": "zh"},
    "bn": {"flores": "ben_Beng", "lid": "bn"},
    "te": {"flores": "tel_Telu", "lid": "te"},
}
ENGLISH_FLORES = "eng_Latn"
ENGLISH_LID = "en"
FLORES_SPLITS = ("dev", "devtest")


LAYER_STRIDE = 4            
N_SENTENCES = 300           
N_CALIBRATION = 24          
HORIZON_T = 3               
TOP_CANDIDATES_PER_LAYER = 150   
TOP_K_FEATURES_PER_LAYER = 20    
INTERVENTION_EPS = 8.0           

MAX_BATCH_SIZE = 128

OUTPUT_DIR = "foxp2_stage1_outputs"