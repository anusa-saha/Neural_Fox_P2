"""
Stage I: Localize language-selective SAE features (paper Sec. 2.1, 2.1.1, 2.1.2, 2.1.3).

Given a *pretrained* SAE per layer (loaded via sae_utils.get_sae_for_layer --
no SAE training) and matched (English, target-language) prompt pairs, we:

  1. Compute per-layer feature activations a_j^(l)(x) = z^(l)(x) for every
     matched prompt (Sec. 2.1, Eq.
       z^(l)(x) = ReLU(W_l^T h^(l)(x) + b_l)
     -- here W_l/b_l are a public SAE checkpoint's encoder weights).
  2. Score target-selectivity Sel_j (Sec. 2.1.1):
       Sel_j     = E_k[a_j(x_tgt)] - E_k[a_j(x_en)]
       Sel_hat_j = Sel_j / (Std_k[a_j(x_tgt)] + Std_k[a_j(x_en)] + eps)
  3. Score causal logit-mass lift via micro-interventions (Sec. 2.1.2):
       z_j <- z_j + alpha * e_j  (equivalently, in hidden space, h <- h + alpha * W_dec[j],
                                   since the SAE decoder is linear)
       Lift_j(alpha) = E_x[ Delta_M(x; do(z_j += alpha)) - Delta_M(x) ]
       LiftSlope_j   = median_alpha( Lift_j(alpha) / alpha )
  4. Combine into Score_j = max(Sel_hat_j, 0) * max(LiftSlope_j, 0) and take
     the top-K per layer (Sec. 2.1.3), forming the language-neuron set N_lt.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

from .sae_utils import SimpleSAE
from .activations import ResidualSteer, capture_hidden_states_batched
from .metrics import next_token_distributions, delta_m
from .gpu_utils import free_memory


@dataclass
class Stage1Config:
    candidate_layers: List[int]
    top_k_per_layer: int = 8
    alphas: Tuple[float, ...] = (2.0, 4.0, 8.0)   # multi-magnitude probes for LiftSlope (Sec 2.1.2)
    horizon: int = 3                              # T in {1, 2, 3}
    eps: float = 1e-6
    lift_candidate_pool: int = 64                 # pre-filter size before the (expensive) lift probe
    prompt_batch_size: int = 16                   # matched-pair activation-capture batch size
    lift_probe_batch_size: int = 32               # weak-prompt batch size for the causal-lift decode


@dataclass
class LanguageFeature:
    layer: int
    index: int
    selectivity: float
    lift_slope: float
    score: float


def compute_feature_activations(saes: Dict[int, SimpleSAE], hiddens: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """z^(l)(x) for a batch of hidden states h^(l)(x), one SAE per layer.

    Hidden states may live on CPU (they're captured there by default to keep
    peak VRAM low during long activation-capture loops -- see
    activations.capture_hidden_states_batched); this moves each layer's
    tensor onto whichever device its SAE lives on before encoding.
    """
    out = {}
    for l, h in hiddens.items():
        sae = saes[l]
        if h.device != sae.W_enc.device:
            h = h.to(sae.W_enc.device)
        out[l] = sae.encode(h)
    return out


def selectivity_scores(z_tgt: torch.Tensor, z_en: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Sel_hat_j (Sec. 2.1.1, "Matched-pair selectivity")."""
    mu = z_tgt.mean(0) - z_en.mean(0)
    denom = z_tgt.std(0) + z_en.std(0) + eps
    return mu / denom


@torch.no_grad()
def causal_lift_scores(
    model, tokenizer, model_cfg, layer: int, sae: SimpleSAE,
    feature_indices: torch.Tensor, weak_prompts: List[str],
    target_ids: torch.Tensor, english_ids: torch.Tensor,
    cfg: Stage1Config, device,
) -> torch.Tensor:
    """LiftSlope_j for a set of candidate features at one layer (Sec. 2.1.2).

    Adding alpha to SAE feature j and decoding linearly is equivalent to
    adding alpha * W_dec[j] directly to the residual stream (no encode/decode
    round-trip is needed for this constant-direction probe), which is exactly
    what sae_utils.SimpleSAE.decoder_row(j) is built to hand back
    ("dh/dz_j = W_dec[j]").
    """
    base_probs = next_token_distributions(
        model, tokenizer, weak_prompts, cfg.horizon, device, batch_size=cfg.lift_probe_batch_size,
    )
    base_dm = delta_m(base_probs, target_ids.to(device), english_ids.to(device)).mean(dim=1)  # [B]

    slopes = torch.zeros(len(feature_indices))
    for fi, j in enumerate(feature_indices.tolist()):
        w_dec_j = sae.decoder_row(j).to(device)
        ratios = []
        for alpha in cfg.alphas:
            direction = alpha * w_dec_j

            def fn(h, direction=direction):
                return direction.to(h.dtype)

            with ResidualSteer(model, model_cfg, {layer: fn}):
                probs = next_token_distributions(
                    model, tokenizer, weak_prompts, cfg.horizon, device, batch_size=cfg.lift_probe_batch_size,
                )
            dm = delta_m(probs, target_ids.to(device), english_ids.to(device)).mean(dim=1)
            lift = (dm - base_dm).mean().item()
            ratios.append(lift / alpha)
        ratios.sort()
        slopes[fi] = ratios[len(ratios) // 2]  # median over alpha (Sec 2.1.2)
        if (fi + 1) % 16 == 0:
            free_memory()  # periodically hand fragmented blocks back to the allocator
    return slopes


def rank_and_select(layer: int, sel: torch.Tensor, lift_slope: torch.Tensor, top_k: int) -> List[LanguageFeature]:
    """Score_j = max(Sel_hat_j, 0) * max(LiftSlope_j, 0); top-K per layer (Sec. 2.1.3)."""
    s = sel.clamp(min=0)
    c = lift_slope.clamp(min=0)
    score = s * c
    k = min(top_k, int((score > 0).sum().item()))
    if k == 0:
        return []
    top_idx = torch.topk(score, k).indices
    return [
        LanguageFeature(
            layer=layer, index=int(i), selectivity=float(sel[i]),
            lift_slope=float(lift_slope[i]), score=float(score[i]),
        )
        for i in top_idx
    ]


def localize_language_features(
    model, tokenizer, model_cfg, saes: Dict[int, SimpleSAE],
    en_prompts: List[str], tgt_prompts: List[str], weak_prompts: List[str],
    target_ids: torch.Tensor, english_ids: torch.Tensor,
    cfg: Stage1Config, device,
) -> Dict[int, List[LanguageFeature]]:
    """Full Stage I pipeline -> N_lt organized as {layer: [LanguageFeature, ...]}."""
    h_en = capture_hidden_states_batched(
        model, tokenizer, model_cfg, cfg.candidate_layers, en_prompts,
        batch_size=cfg.prompt_batch_size, device=device, store_device="cpu",
    )
    h_tgt = capture_hidden_states_batched(
        model, tokenizer, model_cfg, cfg.candidate_layers, tgt_prompts,
        batch_size=cfg.prompt_batch_size, device=device, store_device="cpu",
    )
    free_memory()

    z_en = compute_feature_activations(saes, h_en)
    z_tgt = compute_feature_activations(saes, h_tgt)

    selected: Dict[int, List[LanguageFeature]] = {}
    for layer in cfg.candidate_layers:
        sel = selectivity_scores(z_tgt[layer], z_en[layer], cfg.eps)

        # Pre-filter to a manageable candidate pool by selectivity before the
        # expensive, decoding-based causal-lift probe. This does not change
        # the final ranking: Score_j = 0 whenever Sel_hat_j <= 0, so anything
        # outside the top-selectivity pool would score 0 anyway.
        pool = min(cfg.lift_candidate_pool, sel.numel())
        cand_idx = torch.topk(sel.clamp(min=0), pool).indices

        lift = causal_lift_scores(
            model, tokenizer, model_cfg, layer, saes[layer], cand_idx,
            weak_prompts, target_ids, english_ids, cfg, device,
        )
        sel_cand = sel[cand_idx]
        feats = rank_and_select(layer, sel_cand, lift, cfg.top_k_per_layer)
        for f in feats:
            f.index = int(cand_idx[f.index])  # remap local (pool) index -> global feature id
        selected[layer] = feats
    return selected
