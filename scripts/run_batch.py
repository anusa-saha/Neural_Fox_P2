#!/usr/bin/env python
"""
Batch driver: run Stage I-III (+ optional generation) for a *list* of
(model, language) jobs sequentially on one GPU, fully releasing the model
and all SAEs between jobs (pipe.close()) so job N+1 starts from a clean
memory slate regardless of what job N used.

Usage:
    python scripts/run_batch.py --jobs_file scripts/batch_jobs_example.json \
        --output_root outputs/batch_run

Each job in the jobs file may override any NeuralFOXP2Pipeline.run(...) /
generation argument; anything it doesn't set falls back to the matching
--default_* CLI flag. A failing job is logged (with its traceback) to
<output_root>/batch_summary.json and does NOT abort the rest of the batch.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from neural_foxp2 import NeuralFOXP2Pipeline  # noqa: E402
from neural_foxp2.gpu_utils import recommended_budget, memory_snapshot, free_memory  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="Run a batch of Neural FOXP2 jobs sequentially on one GPU.")
    p.add_argument("--jobs_file", required=True,
                    help="JSON file: a list of job dicts, each with at least "
                         "{\"model_key\": ..., \"lang_code\": ...}. See batch_jobs_example.json.")
    p.add_argument("--output_root", required=True,
                    help="Each job writes to <output_root>/<job_name>/ (job_name defaults to "
                         "'<model_key>-<lang_code>' unless the job sets its own 'name').")
    p.add_argument("--device", default="cuda")
    p.add_argument("--default_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    p.add_argument("--default_n_disc", type=int, default=200)
    p.add_argument("--default_n_calib", type=int, default=40)
    p.add_argument("--default_n_weak", type=int, default=60)
    p.add_argument("--default_n_dev", type=int, default=40)
    p.add_argument("--default_top_k_per_layer", type=int, default=8)
    p.add_argument("--default_k_selection_mode", choices=["fixed", "adaptive"], default="fixed")
    p.add_argument("--default_lift_candidate_pool", type=int, default=64)
    p.add_argument("--default_lam", type=float, default=4.0)
    p.add_argument("--default_beta", type=float, default=4.0)
    p.add_argument("--default_tune_beta_kl", action="store_true")
    p.add_argument("--default_gamma", type=float, default=1.0)
    p.add_argument("--default_prompts_file", type=str, default=None)
    p.add_argument("--default_max_new_tokens", type=int, default=128)
    return p


def run_one_job(job: dict, args) -> dict:
    name = job.get("name", f"{job['model_key']}-{job['lang_code']}")
    output_dir = os.path.join(args.output_root, name)
    os.makedirs(output_dir, exist_ok=True)
    dtype_str = job.get("dtype", args.default_dtype)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]

    t0 = time.time()
    pipe = NeuralFOXP2Pipeline(
        model_key=job["model_key"], lang_code=job["lang_code"], device=args.device,
        candidate_layers=job.get("candidate_layers"), dtype=dtype,
        gpu_budget=recommended_budget(args.device),
    )
    try:
        artifacts = pipe.run(
            n_disc=job.get("n_disc", args.default_n_disc),
            n_calib=job.get("n_calib", args.default_n_calib),
            n_weak=job.get("n_weak", args.default_n_weak),
            n_dev=job.get("n_dev", args.default_n_dev),
            top_k_per_layer=job.get("top_k_per_layer", args.default_top_k_per_layer),
            k_selection_mode=job.get("k_selection_mode", args.default_k_selection_mode),
            lift_candidate_pool=job.get("lift_candidate_pool", args.default_lift_candidate_pool),
            adaptive_k_step=job.get("adaptive_k_step", 2),
            adaptive_k_patience=job.get("adaptive_k_patience", 2),
            adaptive_k_min_gain=job.get("adaptive_k_min_gain", 0.005),
            adaptive_k_alpha=job.get("adaptive_k_alpha", 4.0),
            lam=job.get("lam", args.default_lam),
            beta=job.get("beta", args.default_beta),
            tune_beta_kl=job.get("tune_beta_kl", args.default_tune_beta_kl),
            beta_grid=job.get("beta_grid"),
            kl_gain_target=job.get("kl_gain_target", 0.8),
            kl_budget=job.get("kl_budget"),
            gamma=job.get("gamma", args.default_gamma),
            seed=job.get("seed", 0),
            output_dir=output_dir,
        )

        prompts = list(job.get("prompts", []))
        prompts_file = job.get("prompts_file", args.default_prompts_file)
        if prompts_file:
            with open(prompts_file) as f:
                prompts.extend([line.strip() for line in f if line.strip()])

        gammas = job.get("gammas", [job.get("gamma", args.default_gamma)])
        for g in gammas:
            if prompts:
                pipe.generate_batch(
                    prompts, gamma=g, max_new_tokens=job.get("max_new_tokens", args.default_max_new_tokens),
                    output_dir=output_dir,
                )

        return {
            "name": name, "status": "ok", "window": artifacts.window,
            "output_dir": output_dir, "wall_time_s": round(time.time() - t0, 1),
            "memory_after": memory_snapshot(args.device),
        }
    finally:
        pipe.close()
        free_memory()


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.jobs_file) as f:
        jobs = json.load(f)

    os.makedirs(args.output_root, exist_ok=True)
    summary = []
    for idx, job in enumerate(jobs):
        name = job.get("name", f"{job['model_key']}-{job['lang_code']}")
        print(f"\n[batch] ({idx + 1}/{len(jobs)}) Running job: {name}")
        try:
            result = run_one_job(job, args)
            print(f"[batch] Job {name} OK in {result['wall_time_s']}s. Window = {result['window']}")
        except Exception as e:  # noqa: BLE001 -- a bad job must not abort the whole batch
            result = {"name": name, "status": "error", "error": str(e), "traceback": traceback.format_exc()}
            print(f"[batch] Job {name} FAILED: {e}")
        summary.append(result)

        with open(os.path.join(args.output_root, "batch_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    n_ok = sum(1 for r in summary if r["status"] == "ok")
    print(f"\n[batch] Done: {n_ok}/{len(jobs)} jobs succeeded. Summary written to "
          f"{os.path.join(args.output_root, 'batch_summary.json')}")


if __name__ == "__main__":
    main()
