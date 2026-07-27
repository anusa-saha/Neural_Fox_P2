"""
Stage II: identify low-rank steering directions and choose the intervention
window (paper Sec. 2.2).

  1. Restrict paired language-shift activations to the Stage-I support N_l
     (Sec. 2.2, "Language-Shift activation"):
       Delta_z_tilde_k = Pi_{N_l}( z(x_tgt_k) - z(x_en_k) )
  2. Stack into Delta_Z^(l) and take a layerwise SVD:
       Delta_Z^(l) = U^(l) Sigma^(l) (V^(l))^T
  3. Choose steering rank r_l via effective-rank + eigengap diagnostics.
  4. Score each candidate layer by spectral Mass, functional Gain, and
     bootstrap Stability, then select the contiguous window W maximizing
     sum_{l in W} Mass_l * Stab_l (ties broken by Gain).
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import torch

from .sae_utils import SimpleSAE
from .activations import ResidualSteer
from .metrics import next_token_distributions, delta_m


@dataclass
class Stage2Config:
    candidate_layers: List[int]
    window_widths: Tuple[int, ...] = (2, 3, 4, 5)
    bootstrap_rounds: int = 8
    gain_eta: float = 4.0     # steering magnitude used only to *probe* per-layer gain (Sec 2.2)
    horizon: int = 3
    lift_probe_batch_size: int = 32   # weak-prompt batch size for the per-layer gain probe


@dataclass
class LayerGeometry:
    layer: int
    support: torch.Tensor           # indices in N_l (subset of the SAE feature dim m)
    singular_values: torch.Tensor   # sigma_1 >= sigma_2 >= ... (Sec 2.2 SVD)
    directions: torch.Tensor        # V[:, :r_l], shape [m, r_l] (right singular vectors)
    rank: int                       # r_l
    mass: float                     # spectral mass captured by top-r_l (Sec 2.2 "Spectral Strength")
    stability: float                # bootstrap principal-angle stability (Sec 2.2 "Stability")
    gain: float                     # single-layer defaultness-gain probe (Sec 2.2 "Functional sensitivity")


def _support_mask(m: int, support_idx: torch.Tensor, device=None) -> torch.Tensor:
    mask = torch.zeros(m, dtype=torch.bool, device=device)
    mask[support_idx.to(mask.device)] = True
    return mask


def build_shift_matrix(z_tgt: torch.Tensor, z_en: torch.Tensor, support_idx: torch.Tensor) -> torch.Tensor:
    """Delta_Z^(l), restricted to the Stage-I support N_l (Sec. 2.2)."""
    m = z_tgt.shape[-1]
    mask = _support_mask(m, support_idx, device=z_tgt.device)
    return (z_tgt - z_en) * mask.unsqueeze(0)  # [N, m]


def svd_geometry(delta_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Layerwise SVD (Sec 2.2): Delta_Z = U diag(sigma) V^T. Returns (sigma, V)."""
    U, S, Vh = torch.linalg.svd(delta_z.float(), full_matrices=False)
    V = Vh.T  # [m, r], columns are right singular vectors v_i
    return S, V


def effective_rank(sigma: torch.Tensor) -> float:
    """r_eff = exp(-sum p_i log p_i), p_i = sigma_i^2 / sum_j sigma_j^2 (Sec 2.2)."""
    p = sigma ** 2
    p = p / (p.sum() + 1e-12)
    p = p.clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def eigengap_rank(sigma: torch.Tensor, r_eff: float) -> int:
    """r_l = min(ceil(r_eff), i*), i* = argmax_i sigma_i / sigma_{i+1} (Sec 2.2)."""
    if sigma.numel() < 2:
        return max(1, sigma.numel())
    ratios = sigma[:-1] / (sigma[1:] + 1e-12)
    i_star = int(torch.argmax(ratios)) + 1  # ratio index i corresponds to rank i (1-indexed)
    r_init = int(torch.ceil(torch.tensor(float(r_eff))).item())
    return max(1, min(r_init, i_star, sigma.numel()))


def spectral_mass(sigma: torch.Tensor, r: int) -> float:
    """Mass_l = sum_{i<=r} sigma_i^2 / sum_j sigma_j^2 (Sec 2.2 "Spectral Strength")."""
    total = (sigma ** 2).sum()
    top = (sigma[:r] ** 2).sum()
    return float(top / (total + 1e-12))


def bootstrap_stability(delta_z: torch.Tensor, r: int, rounds: int, seed: int = 0) -> float:
    """Stab_l = median_{b != b'} tr(P_b P_b') / r via bootstrap-resampled
    subspace overlap (Sec 2.2 "Stability")."""
    n = delta_z.shape[0]
    if n < 2:
        return 0.0
    rng = random.Random(seed)
    subspaces = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        _, V = svd_geometry(delta_z[idx])
        r_b = min(r, V.shape[1])
        subspaces.append(V[:, :r_b])
    overlaps = []
    for i in range(len(subspaces)):
        for j in range(i + 1, len(subspaces)):
            Vi, Vj = subspaces[i], subspaces[j]
            rmin = min(Vi.shape[1], Vj.shape[1])
            if rmin == 0:
                continue
            overlaps.append((Vi[:, :rmin].T @ Vj[:, :rmin]).pow(2).sum().item() / rmin)
    if not overlaps:
        return 0.0
    overlaps.sort()
    return overlaps[len(overlaps) // 2]


def _feature_to_hidden(sae: SimpleSAE, z_vec: torch.Tensor) -> torch.Tensor:
    """decode(z) - decode(0) = z @ W_dec (SAE decoder is linear; Sec 2.3 preliminaries)."""
    return z_vec.to(sae.W_dec.dtype) @ sae.W_dec


@torch.no_grad()
def probe_layer_gain(
    model, tokenizer, model_cfg, layer: int, direction_hidden: torch.Tensor,
    weak_prompts, target_ids, english_ids, eta: float, horizon: int, device,
    batch_size: int = None,
) -> float:
    """Gain_l: apply the (data-derived, unnormalized) target direction at a
    single layer at strength eta, measure induced Delta_M gain (Sec 2.2,
    "Functional sensitivity"). Used only to *score* candidate windows -- the
    final Stage III edit uses its own tuned (lambda_l, beta_l)."""
    base_probs = next_token_distributions(model, tokenizer, weak_prompts, horizon, device, batch_size=batch_size)
    base_dm = delta_m(base_probs, target_ids.to(device), english_ids.to(device)).mean().item()

    d = eta * direction_hidden

    def fn(h, d=d):
        return d.to(h.dtype)

    with ResidualSteer(model, model_cfg, {layer: fn}):
        probs = next_token_distributions(model, tokenizer, weak_prompts, horizon, device, batch_size=batch_size)
    dm = delta_m(probs, target_ids.to(device), english_ids.to(device)).mean().item()
    return dm - base_dm


def compute_layer_geometries(
    model, tokenizer, model_cfg, saes: Dict[int, SimpleSAE],
    support: Dict[int, torch.Tensor],
    z_tgt: Dict[int, torch.Tensor], z_en: Dict[int, torch.Tensor],
    weak_prompts, target_ids, english_ids, cfg: Stage2Config, device,
) -> Dict[int, LayerGeometry]:
    geoms: Dict[int, LayerGeometry] = {}
    for layer in cfg.candidate_layers:
        support_idx = support.get(layer)
        if support_idx is None or support_idx.numel() == 0:
            continue

        dz = build_shift_matrix(z_tgt[layer], z_en[layer], support_idx)
        sigma, V = svd_geometry(dz)
        r_eff = effective_rank(sigma)
        r = eigengap_rank(sigma, r_eff)
        mass_ = spectral_mass(sigma, r)
        stab = bootstrap_stability(dz, r, cfg.bootstrap_rounds)

        v_r = V[:, :r]
        mean_shift = dz.mean(0)
        proj = v_r @ (v_r.T @ mean_shift)             # P_S(mean shift), in feature space
        direction_hidden = _feature_to_hidden(saes[layer], proj)

        gain = probe_layer_gain(
            model, tokenizer, model_cfg, layer, direction_hidden,
            weak_prompts, target_ids, english_ids, cfg.gain_eta, cfg.horizon, device,
            batch_size=cfg.lift_probe_batch_size,
        )
        geoms[layer] = LayerGeometry(
            layer=layer, support=support_idx, singular_values=sigma, directions=v_r,
            rank=r, mass=mass_, stability=stab, gain=gain,
        )
    return geoms


def select_window(geoms: Dict[int, LayerGeometry], cfg: Stage2Config) -> List[int]:
    """W = argmax_W sum_{l in W} Mass_l * Stab_l, ties broken by Gain (Sec 2.2,
    "Choosing the contiguous window")."""
    layers_sorted = sorted(geoms.keys())
    best_window: List[int] = []
    best_score, best_gain = -1.0, -1.0
    for width in cfg.window_widths:
        if width > len(layers_sorted):
            continue
        for start in range(len(layers_sorted) - width + 1):
            window = layers_sorted[start:start + width]
            score = sum(geoms[l].mass * geoms[l].stability for l in window)
            gain = sum(geoms[l].gain for l in window)
            if score > best_score or (score == best_score and gain > best_gain):
                best_window, best_score, best_gain = window, score, gain
    return best_window
