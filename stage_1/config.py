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

# --- Stage I hyperparameters ---
LAYER_STRIDE = 4            # probe every 4th layer instead of all of them
N_SENTENCES = 300           # matched EN/target sentence pairs used for selectivity (Stage I-A)
N_CALIBRATION = 24          # number of weak prompts used for defaultness / causal lift
HORIZON_T = 3               # aggregate Delta_M over t = 1..HORIZON_T (early commitment window)
TOP_CANDIDATES_PER_LAYER = 150   # pre-filter by selectivity before the expensive causal lift
TOP_K_FEATURES_PER_LAYER = 20    # final size of the localized language-neuron set N_lt per layer
INTERVENTION_EPS = 8.0           # z_j <- z_j + EPS, used for the causal micro-intervention

# --- Batching (tuned for a single RTX PRO 6000 96GB) ---
# Everything GPU-bound in this pipeline is done as batched forward passes now,
# and model_utils.py avoids materializing full-sequence logits (the actual
# cause of large OOMs with big-vocab models like Gemma-2's 256k), so this can
# likely go higher than 128 if you have headroom to spare. Kept a bit more
# conservative than a pure "how much fits" number because sequence length
# varies a lot across languages here - Telugu and Bengali tokenize much less
# efficiently than English/Hindi, so the same batch size costs noticeably
# more activation memory for those languages. Lower this if you still hit
# OOM (especially on te/bn), raise it if you have headroom to spare.
MAX_BATCH_SIZE = 128

OUTPUT_DIR = "foxp2_stage1_outputs"