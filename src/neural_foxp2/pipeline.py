"""
End-to-end Neural FOXP2 pipeline: Stage I -> Stage II -> Stage III.

Wires together config.py (model/language registry) and sae_utils.py
(pretrained SAE loading -- Gemma Scope / Llama Scope / Qwen SAE-Res; no SAE
training) with stage1_localize.py, stage2_geometry.py and stage3_steer.py to
reproduce the method of "Neural FOXP2: Language-Specific Steering for
Targeted Language Improvement in LLMs" without training any autoencoders.

If `output_dir` is provided to `.run(...)`, every intermediate artifact is
written out as JSON (see serialization.py):

    run_config.json, stage1_features.json, stage2_geometry.json,
    stage2_directions.pt, stage3_vectors.json, stage3_vectors.pt

and every `.generate(..., output_dir=...)` call appends one record (prompt,
gamma, Delta_M before/after, generated text) to generation_log.json.

Example
-------
    from neural_foxp2 import NeuralFOXP2Pipeline

    pipe = NeuralFOXP2Pipeline(model_key="llama3_1_8b_instruct", lang_code="hi", device="cuda")
    artifacts = pipe.run(n_disc=150, n_calib=40, n_weak=60, lam=4.0, beta=4.0,
                          output_dir="outputs/llama3_1_8b_instruct-hi")
    print("Window:", artifacts.window)

    text = pipe.generate("Tell me about your day.", gamma=1.0, max_new_tokens=64,
                          output_dir="outputs/llama3_1_8b_instruct-hi")
    print(text)

Notes
-----
- Requires network access to Hugging Face (model + SAE checkpoints + FLORES+
  dataset). The math (Stage II SVD/window-selection, Stage III projector
  construction) is covered by network-free unit tests in tests/test_synthetic.py.
- `qwen3_5_9b` is included in config.py but is a placeholder/hypothetical
  entry (see note in README.md).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import torch

from .config import MODELS
from .sae_utils import get_sae_for_layer, SimpleSAE
from .data_utils import load_flores_pairs, build_matched_prompts, build_weak_prompts
from .activations import ResidualCapture
from .metrics import build_token_sets, next_token_distributions, delta_m
from .stage1_localize import Stage1Config, localize_language_features, compute_feature_activations
from .stage2_geometry import Stage2Config, compute_layer_geometries, select_window
from .stage3_steer import compute_steering_vectors, NeuralFOXP2Steerer
from .serialization import save_stage1, save_stage2, save_stage3, save_run_config, append_generation_log

# Single-GPU, large-VRAM cards (e.g. RTX PRO 6000 Blackwell, 96GB) can hold any
# model in config.MODELS plus every SAE plus all activations at once, so this
# pipeline intentionally never shards a model across devices. Two settings do
# matter for that class of card though:
#   - TF32 matmuls: free speedup on Ampere+ (including Blackwell) for the fp32
#     accumulations that show up in the Stage II SVD / metrics code.
#   - attn_implementation: newer GPU architectures (Blackwell = sm_120) are
#     sometimes ahead of the FlashAttention-2 wheel you have installed, in
#     which case attn_implementation="flash_attention_2" fails hard rather
#     than degrading gracefully. "sdpa" (PyTorch's built-in kernels, which
#     include a Blackwell-covered flash-attention backend) is the safe
#     default; pass attn_implementation="flash_attention_2" explicitly once
#     you've confirmed your installed flash-attn build supports sm_120.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class FOXP2Artifacts:
    layers_with_features: Dict
    geometries: Dict
    window: List[int]
    steering_vectors: Dict


class NeuralFOXP2Pipeline:
    def __init__(self, model_key: str, lang_code: str, device: str = "cuda",
                 candidate_layers: Optional[List[int]] = None, dtype=torch.bfloat16,
                 attn_implementation: str = "sdpa"):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model_key {model_key}; see config.MODELS")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import

        self.model_key = model_key
        self.model_cfg = MODELS[model_key]
        self.lang_code = lang_code
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["hf_id"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load straight onto `device` (device_map={"": device} + low_cpu_mem_usage)
        # instead of building the full model on CPU and then `.to(device)`-ing it --
        # the latter briefly holds two full copies of the weights in memory and is
        # pure overhead once everything (model + SAEs + activations) fits on one
        # card, which it does at 96GB for every model in config.MODELS.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["hf_id"],
            torch_dtype=dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).eval()

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
                self.saes[l] = get_sae_for_layer(None, self.model_cfg, l, device=self.device)

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
        if output_dir:
            save_run_config(
                {
                    "model_key": self.model_key, "lang_code": self.lang_code, "device": self.device,
                    "candidate_layers": self.candidate_layers, "n_disc": n_disc, "n_calib": n_calib,
                    "n_weak": n_weak, "top_k_per_layer": top_k_per_layer, "lam": lam, "beta": beta,
                    "gamma": gamma, "seed": seed,
                },
                output_dir,
            )

        self._load_saes(self.candidate_layers)

        # D_disc: used to discover N_lt (Stage I) and Delta_Z / S^(l) (Stage II).
        pairs_disc = load_flores_pairs(self.lang_code, split="dev", n=n_disc, seed=seed)
        # D_cal: small calibration subset used only for the causal-lift and
        # window-gain probes (kept disjoint from D_disc in spirit; Appendix C).
        pairs_calib = load_flores_pairs(self.lang_code, split="devtest", n=n_calib, seed=seed + 1)

        en_prompts, tgt_prompts = build_matched_prompts(pairs_disc)
        weak_prompts_calib = build_weak_prompts(pairs_calib, rng_seed=seed)[:n_weak]

        # ---- Stage I: localize N_lt --------------------------------------
        print("[foxp2] Stage I: localizing language-selective SAE features...")
        s1_cfg = Stage1Config(candidate_layers=self.candidate_layers, top_k_per_layer=top_k_per_layer)
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
        print(f"[foxp2] Stage I done ({sum(len(v) for v in features.values())} features across "
              f"{len(support)} layers).")

        # ---- Activations for Stage II / III (re-encode matched + weak) ---
        layers = list(support.keys())
        with ResidualCapture(self.model, self.model_cfg, layers) as cap_en:
            _ = self.model(**self.tokenizer(en_prompts, return_tensors="pt", padding=True).to(self.device))
            h_en = dict(cap_en.cache)
        with ResidualCapture(self.model, self.model_cfg, layers) as cap_tgt:
            _ = self.model(**self.tokenizer(tgt_prompts, return_tensors="pt", padding=True).to(self.device))
            h_tgt = dict(cap_tgt.cache)
        z_en = compute_feature_activations(self.saes, h_en)
        z_tgt = compute_feature_activations(self.saes, h_tgt)

        weak_prompts_geo = build_weak_prompts(pairs_disc, rng_seed=seed + 2)
        with ResidualCapture(self.model, self.model_cfg, layers) as cap_weak:
            _ = self.model(**self.tokenizer(weak_prompts_geo, return_tensors="pt", padding=True).to(self.device))
            h_weak = dict(cap_weak.cache)
        z_weak = compute_feature_activations(self.saes, h_weak)

        # ---- Stage II: low-rank directions + window ----------------------
        print("[foxp2] Stage II: computing SVD steering geometry + selecting window...")
        s2_cfg = Stage2Config(candidate_layers=layers)
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
        print(f"[foxp2] Stage II done. Selected window W = {window}.")

        # ---- Stage III: signed sparse steering vectors --------------------
        vectors = compute_steering_vectors(geoms, z_tgt, z_en, z_weak, window, lam=lam, beta=beta)
        if output_dir:
            save_stage3(vectors, output_dir)
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

    def generate(self, prompt: str, gamma: Optional[float] = None,
                 output_dir: Optional[str] = None, metric_horizon: int = 3, **gen_kwargs) -> str:
        """Generate steered text for `prompt`. If `output_dir` is given, also
        computes the early-horizon Delta_M defaultness metric with and
        without steering (Sec. 2, T in {1,2,3} by default) and appends a
        record to generation_log.json for later analysis."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        steerer = self.steerer(gamma)

        base_dm = None
        steer_dm = None
        if output_dir:
            base_probs = next_token_distributions(self.model, self.tokenizer, [prompt], metric_horizon, self.device)
            base_dm = delta_m(base_probs, self.target_ids.to(self.device),
                               self.english_ids.to(self.device)).mean().item()

        with steerer:
            if output_dir:
                steer_probs = next_token_distributions(self.model, self.tokenizer, [prompt], metric_horizon, self.device)
                steer_dm = delta_m(steer_probs, self.target_ids.to(self.device),
                                    self.english_ids.to(self.device)).mean().item()
            out = self.model.generate(**enc, **gen_kwargs)

        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        if output_dir:
            append_generation_log(
                {
                    "model_key": self.model_key,
                    "lang_code": self.lang_code,
                    "prompt": prompt,
                    "gamma": steerer.gamma,
                    "window": self.artifacts.window,
                    "delta_m_baseline": base_dm,
                    "delta_m_steered": steer_dm,
                    "delta_m_gain": (steer_dm - base_dm) if (base_dm is not None and steer_dm is not None) else None,
                    "generated_text": text,
                    "gen_kwargs": {k: v for k, v in gen_kwargs.items() if isinstance(v, (int, float, str, bool))},
                },
                output_dir,
            )
        return text