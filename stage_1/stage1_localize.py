"""
Neural FOXP2 - Stage I: Localize Language Neurons in a (pretrained) SAE
Dictionary Basis. Batched version - see model_utils.py for the batching
machinery. Design, for a single RTX PRO 6000 (96GB):

  - Stage I-A (selectivity): all 600 EN/target sentences for a layer are
    encoded in a handful of MAX_BATCH_SIZE-sized batches instead of one
    forward pass per sentence.
  - Stage I-B (causal lift): instead of looping "for each of 150 candidate
    features: for each of 24 calibration prompts: 3 sequential forward
    passes", we build ONE (candidates x calibration) cross-product - up to
    150*24=3600 rows - each row carrying its own per-feature residual-stream
    delta, and let batched_horizon_defaultness chunk that by MAX_BATCH_SIZE
    and run it through the model. This turns ~10,800 tiny memory-bound
    forward calls per layer into ~45 large, mostly compute-bound ones.
  - The "no edit" baseline is computed once per (model, language) - not
    once per layer - since a zero delta is mathematically a no-op.

Run with:  python stage1_localize.py
"""
import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODELS, LANGUAGES, ENGLISH_FLORES, ENGLISH_LID, FLORES_SPLITS,
    N_SENTENCES, N_CALIBRATION, HORIZON_T, TOP_CANDIDATES_PER_LAYER,
    TOP_K_FEATURES_PER_LAYER, INTERVENTION_EPS, LAYER_STRIDE, MAX_BATCH_SIZE,
    OUTPUT_DIR,
)
from data_utils import load_matched_pairs
from token_sets import build_token_masks
from sae_utils import get_sae_for_layer
from model_utils import (
    get_layer_module, get_last_token_activations_batch,
    batched_horizon_defaultness, prepare_tokenizer_for_batching,
)
from lid_utils import build_lid_identifier, lid_target_minus_english
from weak_prompts import WEAK_PROMPTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_lid(lid_identifier, continuations, target_lid_code):
    return float(np.mean([
        lid_target_minus_english(lid_identifier, c, target_lid_code, ENGLISH_LID)
        for c in continuations
    ]))


def load_model(cfg):
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=DEVICE)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], attn_implementation="sdpa", **kwargs)
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **kwargs)
    model.eval()
    return model


def run_stage1_for_model(model_key, languages=None, layers_override=None):
    """
    languages: list of language keys (subset of LANGUAGES) to run, or None for all.
    layers_override: explicit list of layer indices to probe, or None to use the
    config default (a model's explicit "layers" list if it has one, otherwise
    the LAYER_STRIDE-based range). Out-of-range indices for this model are
    dropped with a warning rather than failing.
    """
    cfg = MODELS[model_key]
    print(f"\n=== {model_key} ({cfg['hf_id']}) ===")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    prepare_tokenizer_for_batching(tokenizer)
    model = load_model(cfg)

    # Some SAE releases only cover specific layers (e.g. Gemma Scope's -it
    # release only has canonical SAEs at 3 layers) - use that exact list if
    # the model config provides one, otherwise fall back to a stride. A CLI
    # --layers override, if given, takes priority over both.
    if layers_override is not None:
        layers = [l for l in layers_override if 0 <= l < cfg["n_layers"]]
        dropped = [l for l in layers_override if l not in layers]
        if dropped:
            print(f"   [warning] layers {dropped} are out of range for {model_key} "
                  f"(n_layers={cfg['n_layers']}) - skipping them.")
    else:
        layers = cfg.get("layers") or list(range(0, cfg["n_layers"], LAYER_STRIDE))

    language_keys = languages if languages is not None else list(LANGUAGES.keys())

    out_root = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(out_root, exist_ok=True)

    calib_texts = WEAK_PROMPTS[:N_CALIBRATION]
    n_calib = len(calib_texts)

    for lang_key in language_keys:
        lang_cfg = LANGUAGES[lang_key]
        print(f"-- language: {lang_key} --")
        pairs = load_matched_pairs(lang_cfg["flores"], N_SENTENCES, ENGLISH_FLORES, FLORES_SPLITS)
        en_texts = [p[0] for p in pairs]
        tgt_texts = [p[1] for p in pairs]

        target_mask, english_mask = build_token_masks(tokenizer, lang_key, device=DEVICE)
        lid_identifier = build_lid_identifier([lang_cfg["lid"], ENGLISH_LID])

        lang_dir = os.path.join(out_root, lang_key)
        os.makedirs(lang_dir, exist_ok=True)

        # --- baseline (no edit), computed ONCE per language, both channels ---
        baseline_mass_rows, baseline_conts = batched_horizon_defaultness(
            model, tokenizer, calib_texts, DEVICE, HORIZON_T, target_mask, english_mask,
            layer_module=None, delta_vecs=None, max_batch_size=MAX_BATCH_SIZE,
        )
        baseline_mass = float(np.mean(baseline_mass_rows))
        baseline_lid = mean_lid(lid_identifier, baseline_conts, lang_cfg["lid"])
        print(f"   baseline (no edit): mass={baseline_mass:.4f}  lid={baseline_lid:.4f}")

        layer_summaries = {}
        for layer in layers:
            print(f"   layer {layer} ...")
            layer_module = get_layer_module(model, cfg["hook_module_path"], layer)
            sae = get_sae_for_layer(model_key, cfg, layer, device=DEVICE)

            # --- Stage I-A: selectivity, batched activation extraction ---
            acts_en = get_last_token_activations_batch(
                model, tokenizer, en_texts, layer_module, DEVICE, MAX_BATCH_SIZE)
            acts_tgt = get_last_token_activations_batch(
                model, tokenizer, tgt_texts, layer_module, DEVICE, MAX_BATCH_SIZE)
            z_en = sae.encode(acts_en.to(DEVICE)).cpu()
            z_tgt = sae.encode(acts_tgt.to(DEVICE)).cpu()

            diff = z_tgt - z_en
            selectivity = diff.mean(0) / (diff.std(0) + 1e-6)  # Sel_j

            n_candidates = min(TOP_CANDIDATES_PER_LAYER, selectivity.numel())
            candidate_idx = torch.topk(selectivity, n_candidates).indices  # tensor

            # --- Stage I-B: causal lift, ONE batched (candidates x calib) cross-product ---
            delta_per_candidate = (INTERVENTION_EPS * sae.decoder_row(candidate_idx)).to(DEVICE)  # [n_candidates, hidden]
            tiled_texts = [t for _ in range(n_candidates) for t in calib_texts]           # candidate-major
            tiled_delta = delta_per_candidate.repeat_interleave(n_calib, dim=0)            # matches tiling order

            mass_rows, _conts = batched_horizon_defaultness(
                model, tokenizer, tiled_texts, DEVICE, HORIZON_T, target_mask, english_mask,
                layer_module=layer_module, delta_vecs=tiled_delta, max_batch_size=MAX_BATCH_SIZE,
            )
            mass_grid = np.array(mass_rows).reshape(n_candidates, n_calib)
            candidate_mean_mass = mass_grid.mean(axis=1)
            lift = {
                int(candidate_idx[i]): float(candidate_mean_mass[i] - baseline_mass)
                for i in range(n_candidates)
            }

            # --- Stage I-C: rank and select N_lt ---
            sel_lookup = {int(j): float(selectivity[j]) for j in candidate_idx}
            score = {j: max(sel_lookup[j], 0.0) * max(lift[j], 0.0) for j in sel_lookup}
            ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
            top_features = [j for j, _ in ranked[:TOP_K_FEATURES_PER_LAYER]]

            # --- construct-validity check: ONE combined edit, both channels ---
            top_idx_tensor = torch.tensor(top_features, device=DEVICE)
            combined_delta = (INTERVENTION_EPS * sae.decoder_row(top_idx_tensor)).sum(dim=0).to(DEVICE)
            combined_mass_rows, combined_conts = batched_horizon_defaultness(
                model, tokenizer, calib_texts, DEVICE, HORIZON_T, target_mask, english_mask,
                layer_module=layer_module, delta_vecs=combined_delta, max_batch_size=MAX_BATCH_SIZE,
            )
            combined_mass = float(np.mean(combined_mass_rows))
            combined_lid = mean_lid(lid_identifier, combined_conts, lang_cfg["lid"])
            gain_mass = combined_mass - baseline_mass
            gain_lid = combined_lid - baseline_lid
            channels_agree = (gain_mass > 0) == (gain_lid > 0)

            layer_summaries[layer] = {
                "selectivity": sel_lookup,
                "lift": lift,
                "score": score,
                "top_features": top_features,
                "combined_edit_gain_mass": gain_mass,
                "combined_edit_gain_lid": gain_lid,
                "channels_agree": channels_agree,
            }
            print(f"      top-{TOP_K_FEATURES_PER_LAYER} combined edit: "
                  f"gain_mass={gain_mass:.4f} gain_lid={gain_lid:.4f} agree={channels_agree}")

            # Save paired feature activations for the *selected* features only -
            # direct input to Stage II's per-layer SVD of Delta Z.
            np.savez(
                os.path.join(lang_dir, f"layer{layer}_features.npz"),
                feature_ids=np.array(top_features, dtype=np.int64),
                z_en=z_en[:, top_features].numpy(),
                z_tgt=z_tgt[:, top_features].numpy(),
            )

        with open(os.path.join(lang_dir, "stage1_summary.json"), "w") as f:
            json.dump({
                "model": model_key,
                "language": lang_key,
                "horizon_T": HORIZON_T,
                "baseline_default_mass": baseline_mass,
                "baseline_default_lid": baseline_lid,
                "layers": {str(l): v for l, v in layer_summaries.items()},
            }, f, indent=2)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage I: localize language-neurons in pretrained SAE dictionaries."
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help=f"Comma-separated model keys to run (e.g. 'gemma2_9b_it,qwen3_8b'). "
             f"Default: all of {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--languages", type=str, default=None,
        help=f"Comma-separated language keys to run (e.g. 'hi,es'). "
             f"Default: all of {list(LANGUAGES.keys())}",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices to probe (e.g. '4,8,12'), overriding "
             "the config default (a model's explicit layer list, or the "
             "LAYER_STRIDE-based range). Applied to every selected model - indices "
             "out of range for a given model are dropped with a warning, so this "
             "is most useful when running one model at a time.",
    )
    args = parser.parse_args()

    def _parse_csv(s):
        return [x.strip() for x in s.split(",") if x.strip()] if s else None

    model_keys = _parse_csv(args.models) or list(MODELS.keys())
    for m in model_keys:
        if m not in MODELS:
            raise ValueError(f"Unknown model key '{m}'. Valid options: {list(MODELS.keys())}")

    language_keys = _parse_csv(args.languages) or list(LANGUAGES.keys())
    for l in language_keys:
        if l not in LANGUAGES:
            raise ValueError(f"Unknown language key '{l}'. Valid options: {list(LANGUAGES.keys())}")

    layers_override = None
    if args.layers:
        layers_override = [int(x) for x in _parse_csv(args.layers)]

    for model_key in model_keys:
        run_stage1_for_model(model_key, languages=language_keys, layers_override=layers_override)
    print("\nDone. Stage I outputs are under:", OUTPUT_DIR)
