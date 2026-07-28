"""
Stage III: signed sparse activation steering targeted to language neurons
(paper Sec. 2.3), plus KL-trust-region tuning of beta (Appendix B).

At every decoding step and only within the Stage-II window W:

    z^(l)(x) <- z^(l)(x) + Pi_N( delta_z+_lt(x) + delta_z-_lt(x) )
    h^(l)(x) <- decode( z^(l)(x) )                              (full reconstruction-replace)

    delta_z+_lt(x) = lambda_l * P_lt^(l) mu_lt^(l)            (constant "push", Sec 2.3a)
    delta_z-_lt(x) = -beta_l * <z(x), mu_en> / (||mu_en||^2 + eps) * mu_en   (state-dependent "suppress", Sec 2.3b)

where P_lt^(l) = Pi_N P_S Pi_N is the composed sparse+low-rank projector,
    mu_lt^(l) = E_k[ Pi_N( z(x_tgt_k) - z(x_en_k) ) ]   (mean target shift, masked)
    mu_en^(l) = E_x~D_weak[ Pi_N z(x) ]                 (English-attractor mean, masked)

The residual stream is REPLACED with the SAE's reconstruction of the edited
feature vector (h <- decode(z + delta_z), Sec 2.3 "Reconstruction and
forward propagation"), not merely shifted by an additive delta on top of the
original h -- these differ by the SAE's own reconstruction error
(h - decode(encode(h))), which is only zero for a perfect autoencoder.

Control intensity (Sec 2.3): (lambda_l, beta_l) <- gamma * (lambda_l, beta_l).

KL-trust-region tuning of beta (Appendix B, "Constrained choice of
kappa(lambda)"): for a fixed lambda, search a grid of beta values and pick
the smallest one that (a) retains at least a target fraction of the grid's
best Delta_M gain and (b) optionally stays within a KL-drift budget --
"the smallest drift solution that still attains the required mechanistic
improvement."
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch

from .sae_utils import SimpleSAE
from .stage2_geometry import LayerGeometry
from .activations import ResidualSteer
from .metrics import next_token_distributions, delta_m


@dataclass
class SteeringVector:
    layer: int
    support_mask: torch.Tensor      # bool [m], Pi_N
    mu_target_final: torch.Tensor   # Pi_N P_S Pi_N mu_lt   -- [m]
    mu_en_masked: torch.Tensor      # Pi_N mu_en             -- [m]
    lam: float                      # lambda_l
    beta: float                     # beta_l


def _mask_vec(m: int, support_idx: torch.Tensor, device) -> torch.Tensor:
    mask = torch.zeros(m, dtype=torch.bool, device=device)
    mask[support_idx.to(device)] = True
    return mask


def compute_steering_vectors(
    geoms: Dict[int, LayerGeometry],
    z_tgt: Dict[int, torch.Tensor], z_en: Dict[int, torch.Tensor],
    z_weak: Dict[int, torch.Tensor],
    window: List[int], lam: float, beta: float, eps: float = 1e-6,
) -> Dict[int, SteeringVector]:
    """Precompute (mu_lt, mu_en) per window layer (Sec 2.3a/b). These are
    data-derived constants; only the negative term's projection coefficient
    <z(x), mu_en> is recomputed at every generation step (see
    NeuralFOXP2Steerer)."""
    vectors: Dict[int, SteeringVector] = {}
    for layer in window:
        geo = geoms[layer]
        m = z_tgt[layer].shape[-1]
        device = z_tgt[layer].device
        mask = _mask_vec(m, geo.support, device)

        mu_target_masked = ((z_tgt[layer] - z_en[layer]) * mask).mean(0)   # Pi_N(mean shift)
        V = geo.directions.to(device)                                      # [m, r]
        proj = V @ (V.T @ mu_target_masked)                                 # P_S(.)
        mu_target_final = mask * proj                                      # Pi_N P_S Pi_N mu_lt

        mu_en_masked = (z_weak[layer] * mask).mean(0)                      # Pi_N mu_en

        vectors[layer] = SteeringVector(
            layer=layer, support_mask=mask,
            mu_target_final=mu_target_final, mu_en_masked=mu_en_masked,
            lam=lam, beta=beta,
        )
    return vectors


class NeuralFOXP2Steerer:
    """Registers Stage-III forward hooks implementing the signed sparse edit.

    Usage:
        steerer = NeuralFOXP2Steerer(model, model_cfg, saes, vectors, gamma=1.0)
        with steerer:
            model.generate(**enc, max_new_tokens=64)

    `gamma` is the paper's single control-intensity knob (Sec 2.3,
    "Control intensity"): (lambda_l, beta_l) <- gamma * (lambda_l, beta_l).
    """

    def __init__(self, model, model_cfg, saes: Dict[int, SimpleSAE],
                 vectors: Dict[int, SteeringVector], gamma: float = 1.0, eps: float = 1e-6):
        self.model = model
        self.model_cfg = model_cfg
        self.saes = saes
        self.vectors = vectors
        self.gamma = gamma
        self.eps = eps
        self._hook_ctx = None

    def _make_fn(self, layer: int):
        vec = self.vectors[layer]
        sae = self.saes[layer]
        gamma = self.gamma
        eps = self.eps

        def fn(h: torch.Tensor) -> torch.Tensor:
            # h: [B, T, d_model] (T = full prompt length on the prefill pass,
            # T = 1 on each incremental decode step).
            z = sae.encode(h)                                     # [B, T, m]
            mask = vec.support_mask.to(h.device)
            z_masked = z * mask

            mu_en = vec.mu_en_masked.to(h.device)
            denom = (mu_en @ mu_en) + eps
            coeff = (z_masked @ mu_en) / denom                     # [B, T]
            delta_neg = -(gamma * vec.beta) * coeff.unsqueeze(-1) * mu_en   # [B, T, m]

            delta_pos = (gamma * vec.lam) * vec.mu_target_final.to(h.device)  # [m]

            delta_z = mask * (delta_pos + delta_neg)                # Pi_N(delta+ + delta-)
            z_new = z + delta_z                                     # z^(l)(x) <- z^(l)(x) + delta
            h_new = sae.decode(z_new)                                # h^(l)(x) <- decode(z_new) (full replace)
            return (h_new - h).to(h.dtype)                           # returned as a delta for the ResidualSteer hook

        return fn

    def __enter__(self):
        fns = {l: self._make_fn(l) for l in self.vectors}
        self._hook_ctx = ResidualSteer(self.model, self.model_cfg, fns)
        self._hook_ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._hook_ctx.__exit__(*exc)
        self._hook_ctx = None


# --------------------------------------------------------------------------
# KL-trust-region tuning of beta (Appendix B, "Constrained choice of kappa")
# --------------------------------------------------------------------------

def compute_kl_drift(steer_probs: torch.Tensor, base_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL_T(x) = mean over early steps t<=T of KL(P_steered(.|ctx_t) ||
    P_baseline(.|ctx_t)) (Appendix B/D "KL trust region"). `steer_probs` /
    `base_probs`: [B, horizon, vocab]. Returns [B]."""
    p = steer_probs.clamp_min(eps)
    q = base_probs.clamp_min(eps)
    kl_per_step = (p * (p.log() - q.log())).sum(dim=-1)  # [B, horizon]
    return kl_per_step.mean(dim=1)  # [B]


def tune_beta_kl_trust_region(
    model, tokenizer, model_cfg, saes: Dict[int, SimpleSAE],
    geoms: Dict[int, LayerGeometry],
    z_tgt: Dict[int, torch.Tensor], z_en: Dict[int, torch.Tensor], z_weak: Dict[int, torch.Tensor],
    window: List[int], lam: float, beta_grid: Sequence[float],
    dev_prompts: List[str], target_ids: torch.Tensor, english_ids: torch.Tensor,
    horizon: int, gain_target_frac: float, kl_budget: Optional[float], device,
    batch_size: Optional[int] = None,
) -> Tuple[float, List[dict]]:
    """Appendix B, "Constrained choice of kappa(lambda)":

        kappa(lambda) = argmin_kappa E_x~Ddev[KL_T0(x; lambda, kappa)]
                         s.t. E_x[Delta_M_T0(x; lambda, kappa)] >= gamma

    Here `kappa` is realized as beta_l (the suppression strength) with
    lambda held fixed at the user-chosen value. For each beta in
    `beta_grid`, we build the corresponding steering vectors, measure the
    mean Delta_M gain and mean KL drift on `dev_prompts` (a held-out D_dev
    pool, distinct from D_disc/D_cal), then select the smallest beta that
    retains >= `gain_target_frac` of the grid's best gain -- and, if
    `kl_budget` is given, also keeps KL drift within that budget. Returns
    (chosen_beta, per-beta diagnostics) for logging.
    """
    base_probs = next_token_distributions(model, tokenizer, dev_prompts, horizon, device, batch_size=batch_size)
    base_dm = delta_m(base_probs, target_ids.to(device), english_ids.to(device)).mean(dim=1)  # [B]

    diagnostics: List[dict] = []
    for beta in beta_grid:
        vectors = compute_steering_vectors(geoms, z_tgt, z_en, z_weak, window, lam=lam, beta=beta)
        steerer = NeuralFOXP2Steerer(model, model_cfg, saes, vectors, gamma=1.0)
        with steerer:
            steer_probs = next_token_distributions(model, tokenizer, dev_prompts, horizon, device, batch_size=batch_size)
        steer_dm = delta_m(steer_probs, target_ids.to(device), english_ids.to(device)).mean(dim=1)
        gain = (steer_dm - base_dm).mean().item()
        kl = compute_kl_drift(steer_probs, base_probs).mean().item()
        diagnostics.append({"beta": float(beta), "gain": gain, "kl": kl})

    max_gain = max(d["gain"] for d in diagnostics)
    target = gain_target_frac * max_gain if max_gain > 0 else float("-inf")

    candidates = [d for d in diagnostics if d["gain"] >= target]
    if kl_budget is not None:
        within_budget = [d for d in candidates if d["kl"] <= kl_budget]
        if within_budget:
            candidates = within_budget
        # else: nothing meets the KL budget while hitting the gain target;
        # fall through and choose from the gain-qualifying set anyway
        # (diagnostics are logged in full either way, so this is visible).

    if not candidates:
        # Nothing hit the gain target at all (e.g. max_gain <= 0): fall back
        # to whichever grid point achieved the best gain.
        best = max(diagnostics, key=lambda d: d["gain"])
        return best["beta"], diagnostics

    # Among qualifying candidates: smallest KL drift, ties broken by smallest
    # beta (paper: "the smallest setting that preserves semantic quality
    # while improving target-language defaultness").
    best = min(candidates, key=lambda d: (d["kl"], d["beta"]))
    return best["beta"], diagnostics
