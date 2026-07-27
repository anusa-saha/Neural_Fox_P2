"""
Stage III: signed sparse activation steering targeted to language neurons
(paper Sec. 2.3).

At every decoding step and only within the Stage-II window W:

    z^(l)(x) <- z^(l)(x) + Pi_N( delta_z+_lt(x) + delta_z-_lt(x) )

    delta_z+_lt(x) = lambda_l * P_lt^(l) mu_lt^(l)            (constant "push", Sec 2.3a)
    delta_z-_lt(x) = -beta_l * <z(x), mu_en> / (||mu_en||^2 + eps) * mu_en   (state-dependent "suppress", Sec 2.3b)

where P_lt^(l) = Pi_N P_S Pi_N is the composed sparse+low-rank projector,
    mu_lt^(l) = E_k[ Pi_N( z(x_tgt_k) - z(x_en_k) ) ]   (mean target shift, masked)
    mu_en^(l) = E_x~D_weak[ Pi_N z(x) ]                 (English-attractor mean, masked)

Because the SAE decoder is linear (decode(z) = z @ W_dec + b_dec), any
additive edit in feature space maps to an additive edit in hidden space:

    decode(z + dz) - decode(z) = dz @ W_dec

so we apply the edit directly to the residual stream via a forward hook (no
full encode/decode round trip for the constant push term; the
state-dependent suppress term needs one cheap SAE encode of the *current*
hidden state per step, restricted to the localized support).
"""
from dataclasses import dataclass
from typing import Dict, List
import torch

from .sae_utils import SimpleSAE
from .stage2_geometry import LayerGeometry
from .activations import ResidualSteer


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
            delta_h = delta_z @ sae.W_dec.to(h.device)               # linear decode
            return delta_h.to(h.dtype)

        return fn

    def __enter__(self):
        fns = {l: self._make_fn(l) for l in self.vectors}
        self._hook_ctx = ResidualSteer(self.model, self.model_cfg, fns)
        self._hook_ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._hook_ctx.__exit__(*exc)
        self._hook_ctx = None
