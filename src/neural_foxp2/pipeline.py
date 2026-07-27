"""
End-to-end Neural FOXP2 pipeline: Stage I -> Stage II -> Stage III.

Wires together config.py (model/language registry) and sae_utils.py
(pretrained SAE loading -- Gemma Scope / Llama Scope / Qwen SAE-Res; no SAE
training) with stage1_localize.py, stage2_geometry.py and stage3_steer.py to
reproduce the method of "Neural FOXP2: Language-Specific Steering for
Targeted Language Improvement in LLMs" without training any autoencoders.

GPU memory
----------
By default the pipeline auto-detects available VRAM (gpu_utils.recommended_budget)
and picks batch sizes accordingly -- large batches on an RTX PRO 6000-class
(96 GB) card, smaller ones on modest GPUs, with automatic batch-size halving
on `torch.cuda.OutOfMemoryError` throughout. Concretely:

  - Every activation-capture / scoring forward pass runs under
    `torch.no_grad()` and is chunked to `GPUBudget.prompt_batch_size` /
    `lift_probe_batch_size` (see activations.py / metrics.py).
  - Per-layer SAEs load in bf16 by default (`GPUBudget.sae_dtype`), roughly
    halving their footprint vs. the fp32 checkpoints they ship as.
  - Once Stage II selects the final intervention window, SAEs for every
    *non-window* layer are freed (`GPUBudget.offload_non_window_saes`) --
    only the window layers' SAEs are needed for Stage III generation-time
    steering.
  - `.generate_batch(...)` batches prompts (padded, left-aligned) instead of
    looping one at a time, again with automatic OOM backoff.

If `output_dir` is provided to `.run(...)`, every intermediate artifact is
written out as JSON (see serialization.py):

    run_config.json, stage1_features.json, stage2_geometry.json,
    stage2_directions.pt, stage3_vectors.json, stage3_vectors.pt,
    memory_report.json

and every `.generate(...)` / `.generate_batch(...)` call (with `output_dir`
set) appends one record per prompt to generation_log.json.

Example
-------
    from neural_foxp2 import NeuralFOXP2Pipeline

    pipe = NeuralFOXP2Pipeline(model_key="llama3_1_8b_instruct", lang_code="hi", device="cuda")
    artifacts = pipe.run(n_disc=150, n_calib=40, n_weak=60, lam=4.0, beta=4.0,
                          output_dir="outputs/llama3_1_8b_instruct-hi")
    print("Window:", artifacts.window)

    texts = pipe.generate_batch(["Tell me about your day.", "What's new?"],
                                 gamma=1.0, max_new_tokens=64,
                                 output_dir="outputs/llama3_1_8b_instruct-hi")
    pipe.close()  # fully release model + SAEs + CUDA memory
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import torch

from .config import MODELS
from .sae_utils import get_sae_for_layer, SimpleSAE
from .data_utils import load_flores_pairs, build_matched_prompts, build_weak_prompts
from .metrics import build_token_sets, next_token_distributions, delta_m
from .stage1_localize import Stage1Config, localize_language_features, compute_feature_activations
from .stage2_geometry import Stage2Config, compute_layer_geometries, select_window
from .stage3_steer import compute_steering_vectors, NeuralFOXP2Steerer
from .serialization import (
    save_stage1, save_stage2, save_stage3, save_run_config,
    append_generation_log, save_memory_report,
)
from .gpu_utils import GPUBudget, recommended_budget, free_memory, memory_snapshot


@dataclass
class FOXP2Artifacts:
    layers_with_features: Dict
    geometries: Dict
    window: List[int]
    steering_vectors: Dict


class NeuralFOXP2Pipeline:
    def __init__(self, model_key: str, lang_code: str, device: str = "cuda",
                 candidate_layers: Optional[List[int]] = None, dtype=torch.bfloat16,
                 gpu_budget: Optional[GPUBudget] = None):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model_key {model_key}; see config.MODELS")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import

        self.model_key = model_key
        self.model_cfg = MODELS[model_key]
        self.lang_code = lang_code
        self.device = device
        self.gpu_budget = gpu_budget or recommended_budget(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["hf_id"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["hf_id"], torch_dtype=dtype
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        n_layers = self.model_cfg["n_layers"]
        # Exclude the first/last couple of layers by default (steering the
        # embedding-adjacent or unembedding-adjacent layers is unstable and
        # outside the paper's "low-to-mid layer window" prior, Sec 2.2).
        self.candidate_layers = candidate_layers or list(range(2, n_layers - 2))

        self.target_ids, self.english_ids = build_token_sets(self.tokenizer, lang_code)
        self.saes: Dict[int, SimpleSAE] = {}

    def _load_saes(self, layers: List[int]):
        for l in layers:
            if l not in self.saes:
                self.saes[l] = get_sae_for_layer(
                    None, self.model_cfg, l, device=self.device, dtype=self.gpu_budget.sae_dtype,
                )

    def _maybe_empty_cache(self):
        if self.gpu_budget.empty_cache_between_stages:
            free_memory()

    def run(
        self,
        n_disc: int = 200,
        n_calib: int = 40,
        n_weak: int = 60,
        top_k_per_layer: int = 8,
        lam: float = 4.0,
        beta: float = 4.0,
        gamma: float = 1.0,
        seed: int = 0,
        output_dir: Optional[str] = None,
    ) -> FOXP2Artifacts:
        t_start = time.time()
        mem_report = {"before_run": memory_snapshot(self.device)}

        if output_dir:
            save_run_config(
                {
                    "model_key": self.model_key, "lang_code": self.lang_code, "device": self.device,
                    "candidate_layers": self.candidate_layers, "n_disc": n_disc, "n_calib": n_calib,
                    "n_weak": n_weak, "top_k_per_layer": top_k_per_layer, "lam": lam, "beta": beta,
                    "gamma": gamma, "seed": seed,
                    "gpu_budget": {
                        "prompt_batch_size": self.gpu_budget.prompt_batch_size,
                        "lift_probe_batch_size": self.gpu_budget.lift_probe_batch_size,
                        "generate_batch_size": self.gpu_budget.generate_batch_size,
                        "sae_dtype": str(self.gpu_budget.sae_dtype),
                        "offload_non_window_saes": self.gpu_budget.offload_non_window_saes,
                    },
                },
                output_dir,
            )

        self._load_saes(self.candidate_layers)
        self._maybe_empty_cache()

        # D_disc: used to discover N_lt (Stage I) and Delta_Z / S^(l) (Stage II).
        pairs_disc = load_flores_pairs(self.lang_code, split="dev", n=n_disc, seed=seed)
        # D_cal: small calibration subset used only for the causal-lift and
        # window-gain probes (kept disjoint from D_disc in spirit; Appendix C).
        pairs_calib = load_flores_pairs(self.lang_code, split="devtest", n=n_calib, seed=seed + 1)

        en_prompts, tgt_prompts = build_matched_prompts(pairs_disc)
        weak_prompts_calib = build_weak_prompts(pairs_calib, rng_seed=seed)[:n_weak]

        # ---- Stage I: localize N_lt --------------------------------------
        print("[foxp2] Stage I: localizing language-selective SAE features...")
        s1_cfg = Stage1Config(
            candidate_layers=self.candidate_layers, top_k_per_layer=top_k_per_layer,
            prompt_batch_size=self.gpu_budget.prompt_batch_size,
            lift_probe_batch_size=self.gpu_budget.lift_probe_batch_size,
        )
        features = localize_language_features(
            self.model, self.tokenizer, self.model_cfg, self.saes,
            en_prompts, tgt_prompts, weak_prompts_calib,
            self.target_ids, self.english_ids, s1_cfg, self.device,
        )
        support = {
            l: torch.tensor([f.index for f in feats], dtype=torch.long)
            for l, feats in features.items() if feats
        }
        if not support:
            raise RuntimeError("Stage I found no language-selective features; "
                                "loosen top_k_per_layer / alphas / candidate_layers.")
        if output_dir:
            save_stage1(features, output_dir)
        mem_report["after_stage1"] = memory_snapshot(self.device)
        self._maybe_empty_cache()
        print(f"[foxp2] Stage I done ({sum(len(v) for v in features.values())} features across "
              f"{len(support)} layers).")

        # ---- Activations for Stage II / III (re-encode matched + weak) ---
        layers = list(support.keys())
        from .activations import capture_hidden_states_batched  # local import: avoids a cycle at module import time

        h_en = capture_hidden_states_batched(
            self.model, self.tokenizer, self.model_cfg, layers, en_prompts,
            batch_size=self.gpu_budget.prompt_batch_size, device=self.device, store_device="cpu",
        )
        h_tgt = capture_hidden_states_batched(
            self.model, self.tokenizer, self.model_cfg, layers, tgt_prompts,
            batch_size=self.gpu_budget.prompt_batch_size, device=self.device, store_device="cpu",
        )
        z_en = compute_feature_activations(self.saes, h_en)
        z_tgt = compute_feature_activations(self.saes, h_tgt)

        weak_prompts_geo = build_weak_prompts(pairs_disc, rng_seed=seed + 2)
        h_weak = capture_hidden_states_batched(
            self.model, self.tokenizer, self.model_cfg, layers, weak_prompts_geo,
            batch_size=self.gpu_budget.prompt_batch_size, device=self.device, store_device="cpu",
        )
        z_weak = compute_feature_activations(self.saes, h_weak)
        del h_en, h_tgt, h_weak
        self._maybe_empty_cache()

        # ---- Stage II: low-rank directions + window ----------------------
        print("[foxp2] Stage II: computing SVD steering geometry + selecting window...")
        s2_cfg = Stage2Config(candidate_layers=layers, lift_probe_batch_size=self.gpu_budget.lift_probe_batch_size)
        geoms = compute_layer_geometries(
            self.model, self.tokenizer, self.model_cfg, self.saes, support,
            z_tgt, z_en, weak_prompts_calib, self.target_ids, self.english_ids, s2_cfg, self.device,
        )
        window = select_window(geoms, s2_cfg)
        if not window:
            raise RuntimeError("Stage II could not select a contiguous window; "
                                "check candidate_layers / window_widths.")
        if output_dir:
            save_stage2(geoms, window, output_dir)
        mem_report["after_stage2"] = memory_snapshot(self.device)
        print(f"[foxp2] Stage II done. Selected window W = {window}.")

        # ---- Stage III: signed sparse steering vectors --------------------
        vectors = compute_steering_vectors(geoms, z_tgt, z_en, z_weak, window, lam=lam, beta=beta)
        if output_dir:
            save_stage3(vectors, output_dir)

        # Only the window layers' SAEs are needed from here on (Stage III
        # steering + generation-time encode). Freeing the rest is the single
        # biggest standing-memory win available once discovery is done.
        if self.gpu_budget.offload_non_window_saes:
            dropped = [l for l in list(self.saes.keys()) if l not in window]
            for l in dropped:
                del self.saes[l]
            self._maybe_empty_cache()
            if dropped:
                print(f"[foxp2] Freed SAEs for {len(dropped)} non-window layers "
                      f"(kept {len(window)} for Stage III).")

        mem_report["after_stage3"] = memory_snapshot(self.device)
        if output_dir:
            save_memory_report(mem_report, output_dir)
        print(f"[foxp2] Stage III steering vectors ready ({len(vectors)} layers). "
              f"Total wall time: {time.time() - t_start:.1f}s.")

        self.artifacts = FOXP2Artifacts(
            layers_with_features=features, geometries=geoms, window=window, steering_vectors=vectors,
        )
        self.gamma = gamma
        return self.artifacts

    def steerer(self, gamma: Optional[float] = None) -> NeuralFOXP2Steerer:
        if not hasattr(self, "artifacts"):
            raise RuntimeError("Call .run(...) before requesting a steerer.")
        return NeuralFOXP2Steerer(
            self.model, self.model_cfg, self.saes, self.artifacts.steering_vectors,
            gamma=gamma if gamma is not None else self.gamma,
        )

    @torch.no_grad()
    def _generate_chunk(self, prompts: List[str], gamma: Optional[float],
                         output_dir: Optional[str], metric_horizon: int, **gen_kwargs) -> List[str]:
        """Generate for one already batch-size-bounded chunk of prompts
        (left-padded so batched decoding is correct for a decoder-only
        model), computing Delta_M before/after if `output_dir` is set."""
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            steerer = self.steerer(gamma)

            base_dm = [None] * len(prompts)
            steer_dm = [None] * len(prompts)
            if output_dir:
                base_probs = next_token_distributions(self.model, self.tokenizer, prompts, metric_horizon, self.device)
                base_dm = delta_m(base_probs, self.target_ids.to(self.device),
                                   self.english_ids.to(self.device)).mean(dim=1).tolist()

            with steerer:
                if output_dir:
                    steer_probs = next_token_distributions(self.model, self.tokenizer, prompts, metric_horizon, self.device)
                    steer_dm = delta_m(steer_probs, self.target_ids.to(self.device),
                                        self.english_ids.to(self.device)).mean(dim=1).tolist()
                out = self.model.generate(**enc, **gen_kwargs)

            texts = [
                self.tokenizer.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(len(prompts))
            ]
        finally:
            self.tokenizer.padding_side = original_padding_side

        if output_dir:
            for prompt, text, bdm, sdm in zip(prompts, texts, base_dm, steer_dm):
                append_generation_log(
                    {
                        "model_key": self.model_key,
                        "lang_code": self.lang_code,
                        "prompt": prompt,
                        "gamma": steerer.gamma,
                        "window": self.artifacts.window,
                        "delta_m_baseline": bdm,
                        "delta_m_steered": sdm,
                        "delta_m_gain": (sdm - bdm) if (bdm is not None and sdm is not None) else None,
                        "generated_text": text,
                        "gen_kwargs": {k: v for k, v in gen_kwargs.items() if isinstance(v, (int, float, str, bool))},
                    },
                    output_dir,
                )
        return texts

    def generate_batch(self, prompts: List[str], gamma: Optional[float] = None,
                        output_dir: Optional[str] = None, metric_horizon: int = 3,
                        batch_size: Optional[int] = None, **gen_kwargs) -> List[str]:
        """Generate steered text for a *list* of prompts, batching them
        (padded, left-aligned) instead of looping one at a time, with
        automatic batch-size halving on `torch.cuda.OutOfMemoryError`."""
        bs = max(1, batch_size or self.gpu_budget.generate_batch_size)
        results: List[str] = []
        i = 0
        while i < len(prompts):
            chunk = prompts[i:i + bs]
            try:
                results.extend(self._generate_chunk(chunk, gamma, output_dir, metric_horizon, **gen_kwargs))
                i += bs
            except torch.cuda.OutOfMemoryError:
                free_memory()
                if bs <= self.gpu_budget.min_batch_size:
                    raise
                bs = max(self.gpu_budget.min_batch_size, bs // 2)
        return results

    def generate(self, prompt: str, gamma: Optional[float] = None,
                 output_dir: Optional[str] = None, metric_horizon: int = 3, **gen_kwargs) -> str:
        """Single-prompt convenience wrapper around generate_batch(...)."""
        return self.generate_batch(
            [prompt], gamma=gamma, output_dir=output_dir, metric_horizon=metric_horizon,
            batch_size=1, **gen_kwargs,
        )[0]

    def close(self):
        """Fully release the model, all SAEs, and CUDA memory. Call this
        before loading a different (model, language) pair in the same
        process -- see scripts/run_batch.py for a multi-job driver."""
        if hasattr(self, "model"):
            del self.model
        self.saes.clear()
        free_memory()
