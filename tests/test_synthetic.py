"""
Network-free sanity checks for the Stage II / Stage III math.

Stage I's causal-lift probe inherently requires a real language model +
autoregressive decoding, so it is only exercised end-to-end via pipeline.py
(which needs network access to Hugging Face). Everything checked here --
selectivity scoring, layerwise SVD / effective-rank / eigengap, spectral
mass, bootstrap subspace stability, and the Stage III composed
sparse+low-rank projector -- is pure tensor math and runs anywhere torch does.
"""
import torch

from neural_foxp2.sae_utils import SimpleSAE
from neural_foxp2.stage1_localize import selectivity_scores
from neural_foxp2.stage2_geometry import (
    build_shift_matrix, svd_geometry, effective_rank, eigengap_rank,
    spectral_mass, bootstrap_stability, LayerGeometry, _feature_to_hidden,
)
from neural_foxp2.stage3_steer import compute_steering_vectors


def make_fake_sae(d_model=32, d_sae=64, seed=0) -> SimpleSAE:
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_model, d_sae, generator=g) * 0.05
    b_enc = torch.zeros(d_sae)
    W_dec = torch.randn(d_sae, d_model, generator=g) * 0.05
    b_dec = torch.zeros(d_model)
    return SimpleSAE(W_enc, b_enc, W_dec, b_dec)


def test_selectivity_shapes():
    z_tgt = torch.rand(50, 64)
    z_en = torch.rand(50, 64)
    sel = selectivity_scores(z_tgt, z_en)
    assert sel.shape == (64,)
    # Selectivity should be strongly positive for a feature that only fires
    # on the target-language condition.
    z_tgt2 = torch.zeros(50, 4)
    z_en2 = torch.zeros(50, 4)
    z_tgt2[:, 0] = 5.0 + 0.1 * torch.randn(50)
    z_en2[:, 0] = 0.0 + 0.1 * torch.randn(50)
    sel2 = selectivity_scores(z_tgt2, z_en2)
    assert sel2[0] > 5.0 and sel2[0] > sel2[1:].abs().max()


def test_stage2_geometry_pipeline():
    torch.manual_seed(0)
    n, m = 40, 64
    support_idx = torch.arange(0, 10)
    z_tgt = torch.randn(n, m) * 0.1
    z_en = torch.randn(n, m) * 0.1
    # Inject a real shared direction inside the support so the SVD has signal
    # to recover (otherwise sigma is pure noise and r_eff ~= rank of support).
    shared = torch.zeros(m)
    shared[support_idx] = torch.randn(len(support_idx))
    z_tgt[:, support_idx] += shared[support_idx] * torch.linspace(0.5, 1.5, n).unsqueeze(1)

    dz = build_shift_matrix(z_tgt, z_en, support_idx)
    assert dz.shape == (n, m)
    assert torch.allclose(dz[:, 15], torch.zeros(n))  # outside support -> exactly zero (Pi_N)

    sigma, V = svd_geometry(dz)
    r_eff = effective_rank(sigma)
    r = eigengap_rank(sigma, r_eff)
    assert 1 <= r <= sigma.numel()

    mass_ = spectral_mass(sigma, r)
    assert 0.0 <= mass_ <= 1.0 + 1e-6
    # The injected rank-1 signal should dominate: top component should carry
    # most of the spectral mass.
    assert spectral_mass(sigma, 1) > 0.5

    stab = bootstrap_stability(dz, r, rounds=6)
    assert 0.0 <= stab <= 1.0 + 1e-6

    sae = make_fake_sae(d_model=32, d_sae=m)
    proj = V[:, :r] @ (V[:, :r].T @ dz.mean(0))
    hidden = _feature_to_hidden(sae, proj)
    assert hidden.shape == (32,)


def test_stage3_steering_vectors_are_masked_and_projected():
    torch.manual_seed(0)
    d_model, m = 16, 32
    sae = make_fake_sae(d_model=d_model, d_sae=m)
    support_idx = torch.arange(0, 5)

    geo = LayerGeometry(
        layer=3, support=support_idx,
        singular_values=torch.tensor([3.0, 2.0, 1.0]),
        directions=torch.eye(m)[:, :3], rank=2, mass=0.8, stability=0.9, gain=0.1,
    )
    z_tgt = {3: torch.randn(10, m) * 0.1}
    z_en = {3: torch.randn(10, m) * 0.1}
    z_weak = {3: torch.randn(10, m) * 0.1}

    vectors = compute_steering_vectors({3: geo}, z_tgt, z_en, z_weak, window=[3], lam=2.0, beta=2.0)
    vec = vectors[3]

    assert vec.mu_target_final.shape == (m,)
    assert vec.mu_en_masked.shape == (m,)
    # Pi_N must zero out everything outside the Stage-I support.
    assert torch.all(vec.mu_target_final[5:] == 0)
    assert torch.all(vec.mu_en_masked[5:] == 0)

    # Simulate one forward-hook application by hand (the real hook logic
    # lives in stage3_steer.NeuralFOXP2Steerer._make_fn) and check shapes /
    # that the resulting hidden-space delta is finite and support-consistent.
    h = torch.randn(2, 3, d_model)
    z = sae.encode(h)
    mask = vec.support_mask
    z_masked = z * mask
    mu_en = vec.mu_en_masked
    coeff = (z_masked @ mu_en) / (mu_en @ mu_en + 1e-6)
    delta_neg = -vec.beta * coeff.unsqueeze(-1) * mu_en
    delta_pos = vec.lam * vec.mu_target_final
    delta_z = mask * (delta_pos + delta_neg)
    delta_h = delta_z @ sae.W_dec

    assert delta_h.shape == (2, 3, d_model)
    assert torch.isfinite(delta_h).all()
    # If mu_en were all-zero (degenerate weak-prompt sample), coeff must not
    # blow up thanks to the +eps in the denominator.
    mu_en_zero = torch.zeros_like(mu_en)
    coeff_zero = (z_masked @ mu_en_zero) / (mu_en_zero @ mu_en_zero + 1e-6)
    assert torch.isfinite(coeff_zero).all()


if __name__ == "__main__":
    test_selectivity_shapes()
    test_stage2_geometry_pipeline()
    test_stage3_steering_vectors_are_masked_and_projected()
    print("All synthetic (network-free) sanity checks passed.")
