"""
Command-line entry point: `foxp2-run` (installed via pip), or
`python scripts/run_pipeline.py` (no install needed).

Runs Stage I -> II -> III end-to-end for one (model, language) pair, saves
every intermediate artifact as JSON under --output_dir, then optionally
generates steered text for one or more prompts (optionally sweeping several
gamma values), batching prompts instead of looping one at a time, and
appends Delta_M metrics + generated text to generation_log.json.

Stage I K selection (Sec 2.1.3): --k_selection_mode chooses between a fixed
top-K per layer (--top_k_per_layer used directly) and adaptive lift
saturation (grows K until marginal Delta_M gain plateaus, bounded above by
--top_k_per_layer). Both modes are always available; the flag just picks
which one runs.

Stage III strength (Sec 2.3 / Appendix B): --lam and --beta remain
directly user-tunable as before. Passing --tune_beta_kl additionally tunes
beta via a KL-trust-region grid search instead of using --beta directly
(lambda stays fixed at --lam either way) -- both paths remain available,
selected by the flag.

GPU memory: batch sizes default to gpu_utils.recommended_budget(device),
which auto-detects VRAM and picks large batches on an RTX PRO 6000-class
(96 GB) card, smaller ones on modest GPUs. Every flag below can override a
default explicitly if you want to tune further.
"""
import argparse
import json

import torch

from .pipeline import NeuralFOXP2Pipeline
from .gpu_utils import GPUBudget, recommended_budget, memory_snapshot


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Neural FOXP2 pipeline (Stages I-III) and log every "
                    "intermediate artifact as JSON.",
    )
    p.add_argument("--model_key", required=True,
                    help="Key in config.MODELS, e.g. llama3_1_8b_instruct, gemma2_9b_it, qwen3_8b")
    p.add_argument("--lang_code", required=True,
                    help="Key in config.LANGUAGES, e.g. hi, es, zh, bn, te")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="Base model weight dtype")
    p.add_argument("--output_dir", required=True,
                    help="Directory to write run_config.json / stage*.json / generation_log.json into")

    p.add_argument("--n_disc", type=int, default=200, help="# matched pairs for Stage I/II discovery (D_disc)")
    p.add_argument("--n_calib", type=int, default=40,
                    help="# pairs sampled *from* D_disc for the Stage I causal-lift / Stage II "
                         "window-gain calibration probes (D_cal subset-of-D_disc, Appendix C)")
    p.add_argument("--n_weak", type=int, default=60, help="# weak/neutral prompts (subset of D_cal)")
    p.add_argument("--n_dev", type=int, default=40,
                    help="Size of the held-out D_dev pool (FLORES devtest split), used only when "
                         "--tune_beta_kl is set (Appendix B)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidate_layers", type=str, default=None,
                    help="Comma-separated layer indices, e.g. '4,5,6,7,8'. Default: all but first/last 2.")

    # --- Stage I: K selection (Sec 2.1.3) -------------------------------
    p.add_argument("--k_selection_mode", choices=["fixed", "adaptive"], default="fixed",
                    help="'fixed': use --top_k_per_layer directly. 'adaptive': lift-saturation search -- "
                         "grow K in Score order (joint causal intervention), stop once marginal E[Delta_M] "
                         "gain plateaus, bounded above by --top_k_per_layer. Both modes are always "
                         "available; this flag selects which one runs.")
    p.add_argument("--top_k_per_layer", type=int, default=8,
                    help="Fixed mode: K used directly. Adaptive mode: max K searched.")
    p.add_argument("--lift_candidate_pool", type=int, default=64,
                    help="# top-selectivity features considered before the (expensive) causal-lift probe.")
    p.add_argument("--adaptive_k_step", type=int, default=2,
                    help="[adaptive mode] K increment tested during the lift-saturation search.")
    p.add_argument("--adaptive_k_patience", type=int, default=2,
                    help="[adaptive mode] consecutive 'flat' steps before declaring a plateau.")
    p.add_argument("--adaptive_k_min_gain", type=float, default=0.005,
                    help="[adaptive mode] marginal Delta_M gain below which a step counts as 'flat'.")
    p.add_argument("--adaptive_k_alpha", type=float, default=4.0,
                    help="[adaptive mode] per-feature magnitude used for the joint multi-feature probe.")

    # --- Stage III: steering strength (Sec 2.3 / Appendix B) -----------
    p.add_argument("--lam", type=float, default=4.0, help="Stage III lambda_l (positive push strength)")
    p.add_argument("--beta", type=float, default=4.0,
                    help="Stage III beta_l (English suppression strength). Used directly unless "
                         "--tune_beta_kl is set, in which case it only seeds the default search grid.")
    p.add_argument("--tune_beta_kl", action="store_true",
                    help="Tune beta_l via KL-trust-region search (Appendix B, 'Constrained choice of "
                         "kappa(lambda)') instead of using --beta directly. lambda (--lam) stays fixed "
                         "either way. Both the direct (--beta) and tuned (--tune_beta_kl) paths remain "
                         "available; this flag selects which one runs.")
    p.add_argument("--beta_grid", type=str, default=None,
                    help="Comma-separated beta values to search when --tune_beta_kl is set "
                         "(default: {0.25,0.5,1,2,4,8} x --beta).")
    p.add_argument("--kl_gain_target", type=float, default=0.8,
                    help="[--tune_beta_kl] required fraction of the grid's max Delta_M gain the chosen "
                         "beta must retain.")
    p.add_argument("--kl_budget", type=float, default=None,
                    help="[--tune_beta_kl] optional hard cap on mean KL drift for the chosen beta.")
    p.add_argument("--gamma", type=float, default=1.0, help="Default control-intensity knob for generation")

    # --- GPU memory / batching -----------------------------------------
    p.add_argument("--batch_size", type=int, default=None,
                    help="Override GPUBudget.prompt_batch_size (activation-capture batch). "
                         "Default: auto-detected from VRAM (32 on an RTX PRO 6000-class 96GB card).")
    p.add_argument("--lift_probe_batch_size", type=int, default=None,
                    help="Override GPUBudget.lift_probe_batch_size (causal-lift/gain-probe decode batch).")
    p.add_argument("--generate_batch_size", type=int, default=None,
                    help="Override GPUBudget.generate_batch_size (prompts per batched .generate() call).")
    p.add_argument("--sae_dtype", default=None, choices=["bfloat16", "float16", "float32"],
                    help="Storage dtype for pretrained SAE weights (default: bf16 -- halves SAE VRAM footprint).")
    p.add_argument("--keep_all_saes", action="store_true",
                    help="Keep every candidate layer's SAE in memory after Stage II (default: free "
                         "non-window-layer SAEs, since only the window layers are needed for Stage III).")
    p.add_argument("--no_empty_cache", action="store_true",
                    help="Disable torch.cuda.empty_cache() calls between stages (they're cheap; leave enabled "
                         "unless you're specifically profiling allocator behavior).")

    p.add_argument("--prompt", type=str, default=None, help="Single ad-hoc prompt to generate for")
    p.add_argument("--prompts_file", type=str, default=None, help="Text file, one prompt per line")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--gammas", type=str, default=None,
                    help="Comma-separated gamma values to sweep at generation time, e.g. '0,0.5,1,2'. "
                         "Overrides --gamma for the generation loop (Stage I-III artifacts are unaffected).")
    return p


def _build_gpu_budget(args) -> GPUBudget:
    budget = recommended_budget(args.device)
    if args.batch_size is not None:
        budget.prompt_batch_size = args.batch_size
    if args.lift_probe_batch_size is not None:
        budget.lift_probe_batch_size = args.lift_probe_batch_size
    if args.generate_batch_size is not None:
        budget.generate_batch_size = args.generate_batch_size
    if args.sae_dtype is not None:
        budget.sae_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                             "float32": torch.float32}[args.sae_dtype]
    if args.keep_all_saes:
        budget.offload_non_window_saes = False
    if args.no_empty_cache:
        budget.empty_cache_between_stages = False
    return budget


def main(argv=None):
    args = build_parser().parse_args(argv)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    candidate_layers = [int(x) for x in args.candidate_layers.split(",")] if args.candidate_layers else None
    beta_grid = [float(x) for x in args.beta_grid.split(",")] if args.beta_grid else None
    gpu_budget = _build_gpu_budget(args)

    print(f"[foxp2] GPU memory snapshot before load: {json.dumps(memory_snapshot(args.device))}")
    print(f"[foxp2] Using batch sizes: prompt={gpu_budget.prompt_batch_size} "
          f"lift_probe={gpu_budget.lift_probe_batch_size} generate={gpu_budget.generate_batch_size} "
          f"sae_dtype={gpu_budget.sae_dtype}")
    print(f"[foxp2] K selection: {args.k_selection_mode} (top_k_per_layer={args.top_k_per_layer}); "
          f"beta: {'KL-tuned' if args.tune_beta_kl else 'fixed=' + str(args.beta)}")

    pipe = NeuralFOXP2Pipeline(
        model_key=args.model_key, lang_code=args.lang_code, device=args.device,
        candidate_layers=candidate_layers, dtype=dtype, gpu_budget=gpu_budget,
    )
    artifacts = pipe.run(
        n_disc=args.n_disc, n_calib=args.n_calib, n_weak=args.n_weak, n_dev=args.n_dev,
        top_k_per_layer=args.top_k_per_layer, k_selection_mode=args.k_selection_mode,
        lift_candidate_pool=args.lift_candidate_pool,
        adaptive_k_step=args.adaptive_k_step, adaptive_k_patience=args.adaptive_k_patience,
        adaptive_k_min_gain=args.adaptive_k_min_gain, adaptive_k_alpha=args.adaptive_k_alpha,
        lam=args.lam, beta=args.beta, tune_beta_kl=args.tune_beta_kl, beta_grid=beta_grid,
        kl_gain_target=args.kl_gain_target, kl_budget=args.kl_budget,
        gamma=args.gamma, seed=args.seed, output_dir=args.output_dir,
    )
    print(f"[foxp2] Stage I/II/III complete. Window = {artifacts.window}")

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    gammas = [float(x) for x in args.gammas.split(",")] if args.gammas else [args.gamma]

    for g in gammas:
        if not prompts:
            continue
        texts = pipe.generate_batch(
            prompts, gamma=g, max_new_tokens=args.max_new_tokens, output_dir=args.output_dir,
        )
        for prompt, text in zip(prompts, texts):
            print(f"\n[gamma={g}] {prompt!r}\n-> {text}\n")

    print(f"[foxp2] GPU memory snapshot after run: {json.dumps(memory_snapshot(args.device))}")
    print(f"[foxp2] All artifacts written under {args.output_dir}")
    pipe.close()


if __name__ == "__main__":
    main()
