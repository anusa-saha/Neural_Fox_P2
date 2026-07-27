"""
Command-line entry point: `foxp2-run` (installed via pip), or
`python scripts/run_pipeline.py` (no install needed).

Runs Stage I -> II -> III end-to-end for one (model, language) pair, saves
every intermediate artifact as JSON under --output_dir, then optionally
generates steered text for one or more prompts (optionally sweeping several
gamma values) and appends Delta_M metrics + generated text to
generation_log.json.
"""
import argparse
import time

import torch

from .pipeline import NeuralFOXP2Pipeline


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
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"],
                    help="Attention kernel backend. 'sdpa' (default) uses PyTorch's built-in "
                         "kernels and is safe on new GPU architectures (e.g. Blackwell/sm_120) "
                         "even if your installed flash-attn wheel doesn't yet support them. Only "
                         "pass 'flash_attention_2' once you've confirmed your flash-attn build "
                         "supports your GPU's compute capability.")
    p.add_argument("--output_dir", required=True,
                    help="Directory to write run_config.json / stage*.json / generation_log.json into")

    p.add_argument("--capture_batch_size", type=int, default=16,
                    help="Chunk size for activation-capture forward passes. Lower this if you "
                         "still hit CUDA OOM during Stage I/II activation capture (e.g. on a "
                         "smaller GPU or with very long prompts); raise it for more throughput "
                         "on a large-VRAM card.")
    p.add_argument("--n_disc", type=int, default=200, help="# matched pairs for Stage I/II discovery")
    p.add_argument("--n_calib", type=int, default=40, help="# pairs for causal-lift/window-gain calibration")
    p.add_argument("--n_weak", type=int, default=60, help="# weak/neutral prompts (subset of n_calib pairs)")
    p.add_argument("--top_k_per_layer", type=int, default=8, help="Stage I top-K features per layer")
    p.add_argument("--lam", type=float, default=4.0, help="Stage III lambda_l (positive push strength)")
    p.add_argument("--beta", type=float, default=4.0, help="Stage III beta_l (English suppression strength)")
    p.add_argument("--gamma", type=float, default=1.0, help="Default control-intensity knob for generation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidate_layers", type=str, default=None,
                    help="Comma-separated layer indices, e.g. '4,5,6,7,8'. Default: all but first/last 2.")

    p.add_argument("--prompt", type=str, default=None, help="Single ad-hoc prompt to generate for")
    p.add_argument("--prompts_file", type=str, default=None, help="Text file, one prompt per line")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--gammas", type=str, default=None,
                    help="Comma-separated gamma values to sweep at generation time, e.g. '0,0.5,1,2'. "
                         "Overrides --gamma for the generation loop (Stage I-III artifacts are unaffected).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    candidate_layers = [int(x) for x in args.candidate_layers.split(",")] if args.candidate_layers else None

    t0 = time.time()
    pipe = NeuralFOXP2Pipeline(
        model_key=args.model_key, lang_code=args.lang_code, device=args.device,
        candidate_layers=candidate_layers, dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    artifacts = pipe.run(
        n_disc=args.n_disc, n_calib=args.n_calib, n_weak=args.n_weak,
        top_k_per_layer=args.top_k_per_layer, lam=args.lam, beta=args.beta,
        gamma=args.gamma, seed=args.seed, output_dir=args.output_dir,
        capture_batch_size=args.capture_batch_size,
    )
    print(f"[foxp2] Stage I/II/III complete in {time.time() - t0:.1f}s. Window = {artifacts.window}")

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    gammas = [float(x) for x in args.gammas.split(",")] if args.gammas else [args.gamma]

    for prompt in prompts:
        for g in gammas:
            text = pipe.generate(
                prompt, gamma=g, max_new_tokens=args.max_new_tokens, output_dir=args.output_dir,
            )
            print(f"\n[gamma={g}] {prompt!r}\n-> {text}\n")

    print(f"[foxp2] All artifacts written under {args.output_dir}")


if __name__ == "__main__":
    main()
