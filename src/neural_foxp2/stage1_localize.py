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
       z_j <- z_j + alpha * e_j ;  h(l)(x) <- W_l z(l)(x)   (full SAE reconstruction-replace)
       Lift_j(alpha) = E_x[ Delta_M(x; do(z_j += alpha)) - Delta_M(x) ]
       LiftSlope_j   = median_alpha( Lift_j(alpha) / alpha )
  4. Combine into Score_j = max(Sel_hat_j, 0) * max(LiftSlope_j, 0) and select
     K features per layer (Sec. 2.1.3), forming the language-neuron set N_lt,
     via either:
       - "fixed":    the top `top_k_per_layer` features by Score, or
       - "adaptive": lift-saturation -- grow K in Score order, jointly
         intervening on the top-K features at once, and stop once the
         marginal E[Delta_M] gain plateaus (paper: "We choose K using lift
         saturation, adding features until gains in E[Delta_M_lt] plateau.").
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch

from .sae_utils import SimpleSAE
from .activations import ResidualSteer, capture_hidden_states_batched
from .metrics import next_token_distributions, delta_m
from .gpu_utils import free_memory


@dataclass
class Stage1Config:
    candidate_layers: List[int]
    top_k_per_layer: int = 8                      # fixed mode: K used directly; adaptive mode: max K searched
    alphas: Tuple[float, ...] = (2.0, 4.0, 8.0)   # multi-magnitude probes for LiftSlope (Sec 2.1.2)
    horizon: int = 3                              # T in {1, 2, 3}
    eps: float = 1e-6
    lift_candidate_pool: int = 64                 # pre-filter size before the (expensive) lift probe
    prompt_batch_size: int = 16                   # matched-pair activation-capture batch size
    lift_probe_batch_size: int = 32               # weak-prompt batch size for the causal-lift decode

    # --- K selection (Sec 2.1.3) ---
    k_selection_mode: str = "fixed"               # "fixed" | "adaptive" (lift saturation)
    adaptive_k_step: int = 2                      # K increment tested during the lift-saturation search
    adaptive_k_patience: int = 2                  # consecutive "flat" steps before declaring a plateau
    adaptive_k_min_gain: float = 0.005            # marginal E[Delta_M] gain below which a step counts as "flat"
    adaptive_k_alpha: float = 4.0                 # per-feature magnitude used for the joint multi-feature probe


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

    Implements the paper's literal edit rule: z_j <- z_j + alpha*e_j, then
    h^(l)(x) <- W_l z^(l)(x) (full SAE reconstruction-replace, via
    SimpleSAE.decode -- including the decoder bias, since the actual
    pretrained SAEs used here have one even where the paper's own notation
    omits it). This differs from -- and supersedes -- a pure additive
    `h += alpha * W_dec[j]` edit, which is only equivalent when h already
    equals the SAE's own reconstruction of it (i.e. zero reconstruction
    error), which real SAEs don't achieve exactly.
    """
    base_probs = next_token_distributions(
        model, tokenizer, weak_prompts, cfg.horizon, device, batch_size=cfg.lift_probe_batch_size,
    )
    base_dm = delta_m(base_probs, target_ids.to(device), english_ids.to(device)).mean(dim=1)  # [B]

    slopes = torch.zeros(len(feature_indices))
    for fi, j in enumerate(feature_indices.tolist()):
        ratios = []
        for alpha in cfg.alphas:
            def fn(h, sae=sae, j=j, alpha=alpha):
                z = sae.encode(h)
                z_edit = z.clone()
                z_edit[..., j] = z_edit[..., j] + alpha
                h_new = sae.decode(z_edit)
                return (h_new - h).to(h.dtype)

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


def rank_candidates(layer: int, sel: torch.Tensor, lift_slope: torch.Tensor, max_k: int) -> List[LanguageFeature]:
    """Score_j = max(Sel_hat_j, 0) * max(LiftSlope_j, 0); returns up to
    `max_k` candidates ranked descending by Score, restricted to Score > 0
    (Sec. 2.1.3). Used as-is for "fixed" K, or as the ranked pool that
    "adaptive" K search grows into incrementally."""
    s = sel.clamp(min=0)
    c = lift_slope.clamp(min=0)
    score = s * c
    k = min(max_k, int((score > 0).sum().item()))
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


def find_plateau_k(gains: Sequence[float], k_values: Sequence[int], patience: int, min_gain: float) -> int:
    """Pure plateau-detection logic for lift-saturation K search (Sec 2.1.3),
    separated out from the model-dependent gain computation so it's directly
    unit-testable.

    `gains[i]` is the cumulative E[Delta_M] achieved using the top-`k_values[i]`
    features (k_values assumed increasing). Returns the smallest k at which
    marginal gain has plateaued: the first k such that the next `patience`
    steps all have marginal gain below `min_gain`. If no plateau is detected
    before the schedule ends, returns k_values[-1] (use the full schedule).
    """
    if not k_values:
        return 0
    if len(gains) <= 1:
        return k_values[-1]
    consecutive_flat = 0
    for i in range(1, len(gains)):
        marginal = gains[i] - gains[i - 1]
        if marginal < min_gain:
            consecutive_flat += 1
            if consecutive_flat >= patience:
                return k_values[i - patience]  # the K right before the flat run started
        else:
            consecutive_flat = 0
    return k_values[-1]


@torch.no_grad()
def _joint_intervention_gain(
    model, tokenizer, model_cfg, sae: SimpleSAE, layer: int, feature_indices: List[int], alpha: float,
    weak_prompts: List[str], target_ids: torch.Tensor, english_ids: torch.Tensor,
    horizon: int, device, batch_size: Optional[int] = None,
) -> float:
    """E[Delta_M] gain from jointly adding `alpha` to every feature in
    `feature_indices` at `layer` (full reconstruction-replace), relative to
    the unedited baseline -- the quantity lift-saturation K search grows
    against (Sec. 2.1.3, "adding features until gains ... plateau")."""
    base_probs = next_token_distributions(model, tokenizer, weak_prompts, horizon, device, batch_size=batch_size)
    base_dm = delta_m(base_probs, target_ids.to(device), english_ids.to(device)).mean().item()

    if not feature_indices:
        return 0.0
    idx = torch.as_tensor(feature_indices, dtype=torch.long, device=device)

    def fn(h, sae=sae, idx=idx, alpha=alpha):
        z = sae.encode(h)
        z_edit = z.clone()
        z_edit[..., idx] = z_edit[..., idx] + alpha
        h_new = sae.decode(z_edit)
        return (h_new - h).to(h.dtype)

    with ResidualSteer(model, model_cfg, {layer: fn}):
        probs = next_token_distributions(model, tokenizer, weak_prompts, horizon, device, batch_size=batch_size)
    dm = delta_m(probs, target_ids.to(device), english_ids.to(device)).mean().item()
    return dm - base_dm


def select_features_adaptive_k(
    model, tokenizer, model_cfg, sae: SimpleSAE, layer: int,
    ranked_candidates: List[LanguageFeature], weak_prompts: List[str],
    target_ids: torch.Tensor, english_ids: torch.Tensor, cfg: Stage1Config, device,
) -> List[LanguageFeature]:
    """Lift-saturation K selection (Sec. 2.1.3): incrementally grow the
    feature set (already ranked descending by Score) and stop once marginal
    E[Delta_M] gain plateaus. `cfg.top_k_per_layer` bounds the max K searched."""
    if not ranked_candidates:
        return []
    max_k = min(cfg.top_k_per_layer, len(ranked_candidates))
    k_values = list(range(cfg.adaptive_k_step, max_k + 1, cfg.adaptive_k_step))
    if not k_values or k_values[-1] != max_k:
        k_values.append(max_k)

    gains = []
    for k in k_values:
        idxs = [f.index for f in ranked_candidates[:k]]
        gain = _joint_intervention_gain(
            model, tokenizer, model_cfg, sae, layer, idxs, cfg.adaptive_k_alpha,
            weak_prompts, target_ids, english_ids, cfg.horizon, device,
            batch_size=cfg.lift_probe_batch_size,
        )
        gains.append(gain)

    best_k = find_plateau_k(gains, k_values, cfg.adaptive_k_patience, cfg.adaptive_k_min_gain)
    return ranked_candidates[:best_k]


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
        # expensive, decoding-based causal-lift probe. NOTE: this is a
        # tractability heuristic, not a rank-preserving guarantee -- it's
        # exact only for the Sel_hat_j <= 0 boundary (those are guaranteed
        # Score_j = 0 regardless of lift). A feature with modest positive
        # selectivity but very high LiftSlope could in principle be excluded
        # from the pool -- and thus never scored -- even though its
        # composite Score might exceed a pool member's. Raise
        # `lift_candidate_pool` to reduce this risk, at increased Stage I
        # compute cost (each pooled feature costs len(alphas) extra decodes).
        pool = min(cfg.lift_candidate_pool, sel.numel())
        cand_idx = torch.topk(sel.clamp(min=0), pool).indices

        lift = causal_lift_scores(
            model, tokenizer, model_cfg, layer, saes[layer], cand_idx,
            weak_prompts, target_ids, english_ids, cfg, device,
        )
        sel_cand = sel[cand_idx]
        # Rank the *full* positive-scoring pool first; "fixed" mode then
        # truncates to top_k_per_layer, "adaptive" mode searches within it.
        ranked = rank_candidates(layer, sel_cand, lift, max_k=pool)
        for f in ranked:
            f.index = int(cand_idx[f.index])  # remap local (pool) index -> global feature id

        if cfg.k_selection_mode == "adaptive":
            feats = select_features_adaptive_k(
                model, tokenizer, model_cfg, saes[layer], layer, ranked,
                weak_prompts, target_ids, english_ids, cfg, device,
            )
        else:
            feats = ranked[:cfg.top_k_per_layer]
        selected[layer] = feats
    return selected
