"""
JSON-serialization helpers for Neural FOXP2 intermediate artifacts.

Every stage writes a human-readable JSON file into `output_dir` so results can
be inspected/analyzed without re-running the model:

    output_dir/
      run_config.json          -- every CLI/pipeline argument used for this run
      stage1_features.json     -- Sel_j, LiftSlope_j, Score_j, per layer -> N_lt
      stage2_geometry.json     -- rank r_l, Mass_l, Stability_l, Gain_l, spectrum, + window W
      stage2_directions.pt     -- raw right-singular-vector matrices V[:, :r_l] (torch)
      stage3_vectors.json      -- lambda_l, beta_l, support size, mu norms (summary)
      stage3_vectors.pt        -- raw mu_target_final / mu_en_masked / support_mask (torch)
      generation_log.json      -- one growing list, one entry per .generate() call:
                                  prompt, gamma, Delta_M before/after, generated text

Large tensors (feature/steering vectors, full singular-value spectra) are
saved in full as .pt files for reproducibility, and *previewed* (head/tail +
length) inside the JSON so the JSON files stay small and readable while still
pointing at everything needed for deeper analysis.
"""
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def _tensor_to_preview(t: torch.Tensor, max_len: int = 64):
    """Full list if small enough to be useful inline, else a head/tail preview."""
    flat = t.detach().float().reshape(-1).cpu()
    n = flat.numel()
    if n <= max_len:
        return flat.tolist()
    half = max_len // 2
    return {
        "_truncated": True,
        "length": n,
        "head": flat[:half].tolist(),
        "tail": flat[-half:].tolist(),
    }


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, torch.Tensor):
        return _tensor_to_preview(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def dump_json(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_run_config(config: Dict[str, Any], output_dir: str):
    payload = dict(config)
    payload["_saved_at"] = time.time()
    dump_json(payload, os.path.join(output_dir, "run_config.json"))


def save_stage1(features: Dict[int, List], output_dir: str):
    """Stage I: N_lt -- selectivity, causal-lift-slope, composite score, per layer."""
    payload = {
        str(layer): [dataclasses.asdict(f) for f in feats]
        for layer, feats in features.items()
    }
    n_total = sum(len(v) for v in payload.values())
    dump_json(
        {"n_layers_with_features": len(payload), "n_total_features": n_total, "layers": payload},
        os.path.join(output_dir, "stage1_features.json"),
    )


def save_stage2(geoms: Dict[int, Any], window: List[int], output_dir: str):
    """Stage II: per-layer SVD/window-selection diagnostics + the selected window W."""
    payload = {}
    for layer, g in geoms.items():
        payload[str(layer)] = {
            "layer": g.layer,
            "support_size": int(g.support.numel()),
            "support_indices": g.support.tolist(),
            "rank": g.rank,
            "mass": g.mass,
            "stability": g.stability,
            "gain": g.gain,
            "n_singular_values": int(g.singular_values.numel()),
            "singular_values_preview": _tensor_to_preview(g.singular_values, max_len=64),
            "in_selected_window": layer in window,
        }
    dump_json({"window": window, "layers": payload}, os.path.join(output_dir, "stage2_geometry.json"))
    torch.save(
        {str(l): g.directions.cpu() for l, g in geoms.items()},
        os.path.join(output_dir, "stage2_directions.pt"),
    )


def save_stage3(vectors: Dict[int, Any], output_dir: str):
    """Stage III: per-window-layer steering-vector summary (lambda, beta, norms)."""
    payload = {}
    for layer, v in vectors.items():
        payload[str(layer)] = {
            "layer": v.layer,
            "lambda": v.lam,
            "beta": v.beta,
            "support_size": int(v.support_mask.sum().item()),
            "mu_target_final_norm": float(v.mu_target_final.norm().item()),
            "mu_en_masked_norm": float(v.mu_en_masked.norm().item()),
            "mu_target_final_preview": _tensor_to_preview(v.mu_target_final, max_len=32),
            "mu_en_masked_preview": _tensor_to_preview(v.mu_en_masked, max_len=32),
        }
    dump_json(payload, os.path.join(output_dir, "stage3_vectors.json"))
    torch.save(
        {
            str(l): {
                "mu_target_final": v.mu_target_final.cpu(),
                "mu_en_masked": v.mu_en_masked.cpu(),
                "support_mask": v.support_mask.cpu(),
            }
            for l, v in vectors.items()
        },
        os.path.join(output_dir, "stage3_vectors.pt"),
    )


def append_generation_log(entry: Dict[str, Any], output_dir: str):
    """Append one generation record to generation_log.json (a growing JSON
    list), so long sweeps/runs can be inspected incrementally and safely
    resumed/re-analyzed even if a later prompt in the same run crashes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "generation_log.json")
    log = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    entry = dict(entry)
    entry["_logged_at"] = time.time()
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=_json_default)
