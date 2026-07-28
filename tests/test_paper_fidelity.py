"""
Network-free tests targeting the specific paper-fidelity fixes made after a
code review against the Neural FOXP2 paper:

  1. SimpleSAE.decode() -- the full (affine) SAE reconstruction, needed for
     the "replace the residual stream with decode(z_edited)" edit rule
     (Sec 2.1.2 / Sec 2.3), as opposed to a pure additive hidden-space edit.
  2. NeuralFOXP2Steerer now uses that full reconstruction-replace mechanism,
     not a pure additive delta -- verified against a hand-computed formula,
     and shown to differ from the old (incorrect) additive-only behavior.
  3. Stage II's Gain probe now divides by eta, per the paper's formula.
  4. Stage II's window selection now requires true layer-index contiguity
     (a "window" can no longer silently skip a layer with no Stage I
     support and still be treated as contiguous).
  5. Stage I's lift-saturation (adaptive K) plateau-detection logic.
"""
import torch
import torch.nn as nn

from neural_foxp2.sae_utils import SimpleSAE
from neural_foxp2.stage1_localize import find_plateau_k
from neural_foxp2.stage2_geometry import LayerGeometry, Stage2Config, select_window
from neural_foxp2.stage3_steer import compute_steering_vectors, NeuralFOXP2Steerer


def make_sae_with_bias(d_model=6, d_sae=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_model, d_sae, generator=g) * 0.1
    b_enc = torch.randn(d_sae, generator=g) * 0.01
    W_dec = torch.randn(d_sae, d_model, generator=g) * 0.1
    b_dec = torch.randn(d_model, generator=g) * 0.1  # nonzero -- must be included in decode()
    return SimpleSAE(W_enc, b_enc, W_dec, b_dec)


# --- 1. SimpleSAE.decode() -------------------------------------------------

def test_sae_decode_is_affine_and_includes_bias():
    sae = make_sae_with_bias()
    z = torch.randn(3, 4, 10)
    out = sae.decode(z)
    expected = z.float() @ sae.W_dec.float() + sae.b_dec.float()
    assert torch.allclose(out, expected)
    # Bias must actually matter (not silently dropped).
    zero_bias_decode = z.float() @ sae.W_dec.float()
    assert not torch.allclose(out, zero_bias_decode)


# --- 2. Reconstruction-replace steering (Stage III) ------------------------

def test_steerer_uses_full_reconstruction_replace_not_pure_additive():
    torch.manual_seed(1)
    d_model, m = 6, 10
    sae = make_sae_with_bias(d_model=d_model, d_sae=m)

    support_idx = torch.arange(0, 4)
    geo = LayerGeometry(
        layer=0, support=support_idx, singular_values=torch.tensor([2.0, 1.0]),
        directions=torch.eye(m)[:, :2], rank=2, mass=0.5, stability=0.5, gain=0.0,
    )
    z_tgt = {0: torch.randn(5, m) * 0.1}
    z_en = {0: torch.randn(5, m) * 0.1}
    z_weak = {0: torch.randn(5, m) * 0.1}
    vectors = compute_steering_vectors({0: geo}, z_tgt, z_en, z_weak, [0], lam=2.0, beta=1.5)

    steerer = NeuralFOXP2Steerer(model=None, model_cfg=None, saes={0: sae}, vectors=vectors, gamma=1.0)
    fn = steerer._make_fn(0)

    h = torch.randn(2, 3, d_model)
    delta = fn(h)

    # Hand-computed reference using the paper's edit rule: z <- z + delta_z;
    # h <- decode(z_new) (full reconstruction-replace).
    vec = vectors[0]
    z = sae.encode(h)
    mask = vec.support_mask
    z_masked = z * mask
    mu_en = vec.mu_en_masked
    coeff = (z_masked @ mu_en) / (mu_en @ mu_en + steerer.eps)
    delta_neg = -vec.beta * coeff.unsqueeze(-1) * mu_en
    delta_pos = vec.lam * vec.mu_target_final
    delta_z = mask * (delta_pos + delta_neg)
    z_new = z + delta_z
    expected_delta = sae.decode(z_new) - h

    assert torch.allclose(delta, expected_delta, atol=1e-5)

    # This must genuinely differ from the old pure-additive mechanism
    # (delta_z @ W_dec alone, ignoring reconstruction error + bias) -- i.e.
    # the fix changes behavior, it isn't a no-op relative to the bug.
    old_style_delta = delta_z @ sae.W_dec
    assert not torch.allclose(delta, old_style_delta, atol=1e-4)


# --- 3. Stage II Gain formula: divide by eta --------------------------------

def _make_fake_hookable_model():
    """A real (tiny) nn.Module tree exposing 'model.layers.{i}', so
    ResidualSteer can register/remove a forward hook on it. Since the test
    monkeypatches next_token_distributions entirely, the model's forward()
    is never actually invoked -- this only needs to satisfy the hook
    registration path structurally."""
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(2)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    return FakeModel()


def test_probe_layer_gain_divides_by_eta(monkeypatch):
    import neural_foxp2.stage2_geometry as s2

    model = _make_fake_hookable_model()
    model_cfg = {"hook_module_path": "model.layers.{layer}"}
    sae = make_sae_with_bias(d_model=4, d_sae=4)

    call_count = {"n": 0}

    def fake_next_token_distributions(model, tokenizer, prompts, horizon, device, batch_size=None):
        call_count["n"] += 1
        B, H, V = len(prompts), horizon, 4
        probs = torch.zeros(B, H, V)
        if call_count["n"] % 2 == 1:
            probs[..., 1] = 1.0  # baseline call: all mass on "english" token
        else:
            probs[..., 0] = 1.0  # steered call: all mass on "target" token
        return probs

    def fake_delta_m(probs, tgt_ids, en_ids):
        return probs[..., tgt_ids].sum(-1) - probs[..., en_ids].sum(-1)

    monkeypatch.setattr(s2, "next_token_distributions", fake_next_token_distributions)
    monkeypatch.setattr(s2, "delta_m", fake_delta_m)

    target_ids = torch.tensor([0])
    english_ids = torch.tensor([1])
    eta = 4.0

    gain = s2.probe_layer_gain(
        model, tokenizer=None, model_cfg=model_cfg, sae=sae, layer=0,
        direction_z=torch.zeros(4), weak_prompts=["a", "b"],
        target_ids=target_ids, english_ids=english_ids, eta=eta, horizon=3, device="cpu",
        batch_size=None,
    )
    # baseline Delta_M = 0 - 1 = -1 ; steered Delta_M = 1 - 0 = 1
    # raw (steered - baseline) = 2 ; divided by eta=4 -> 0.5
    assert abs(gain - 0.5) < 1e-6


# --- 4. Stage II window selection requires true layer-index contiguity -----

def _fake_geo(layer, mass=1.0, stability=1.0, gain=0.0):
    return LayerGeometry(
        layer=layer, support=torch.tensor([0]), singular_values=torch.tensor([1.0]),
        directions=torch.eye(1), rank=1, mass=mass, stability=stability, gain=gain,
    )


def test_select_window_skips_windows_with_a_gap_layer():
    # Layers 5, 6 have support; layer 7 does NOT (simulating zero Stage I
    # features found there); layers 8, 9 have support again. A width-4
    # window [5,6,7,8] must never be chosen since layer 7 is missing --
    # even though 5,6,8,9 are adjacent *positions* in sorted(geoms.keys()).
    geoms = {
        5: _fake_geo(5, mass=0.9, stability=0.9),
        6: _fake_geo(6, mass=0.9, stability=0.9),
        8: _fake_geo(8, mass=0.9, stability=0.9),
        9: _fake_geo(9, mass=0.9, stability=0.9),
    }
    cfg = Stage2Config(candidate_layers=list(geoms.keys()), window_widths=(2, 3, 4))
    window = select_window(geoms, cfg)
    # Any valid contiguous window must be entirely within {5,6} or {8,9};
    # it must never span across the missing layer 7.
    assert window in ([5, 6], [8, 9])


def test_select_window_finds_true_contiguous_range():
    geoms = {l: _fake_geo(l, mass=0.5, stability=0.5) for l in [3, 4, 5, 6, 7]}
    # Make layer 5 the standout so a window centered there should win.
    geoms[5] = _fake_geo(5, mass=0.99, stability=0.99)
    geoms[4] = _fake_geo(4, mass=0.9, stability=0.9)
    geoms[6] = _fake_geo(6, mass=0.9, stability=0.9)
    cfg = Stage2Config(candidate_layers=list(geoms.keys()), window_widths=(2, 3))
    window = select_window(geoms, cfg)
    assert window == list(range(min(window), max(window) + 1))  # genuinely contiguous
    assert 5 in window


# --- 5. Adaptive-K plateau detection (pure logic) --------------------------

def test_find_plateau_k_detects_plateau():
    # Gains rise quickly then flatten out from k=6 onward.
    k_values = [2, 4, 6, 8, 10]
    gains =    [0.10, 0.25, 0.30, 0.301, 0.302]
    # marginal(6->8)=0.001, marginal(8->10)=0.001 -- both below min_gain=0.005,
    # patience=2 -> should stop and report k=6 (the point before the flat run).
    k = find_plateau_k(gains, k_values, patience=2, min_gain=0.005)
    assert k == 6


def test_find_plateau_k_uses_full_schedule_if_no_plateau():
    k_values = [2, 4, 6]
    gains = [0.1, 0.3, 0.6]  # always improving a lot
    k = find_plateau_k(gains, k_values, patience=2, min_gain=0.005)
    assert k == 6


def test_find_plateau_k_handles_trivial_inputs():
    assert find_plateau_k([], [], patience=2, min_gain=0.01) == 0
    assert find_plateau_k([0.1], [4], patience=2, min_gain=0.01) == 4


if __name__ == "__main__":
    test_sae_decode_is_affine_and_includes_bias()
    test_steerer_uses_full_reconstruction_replace_not_pure_additive()
    test_select_window_skips_windows_with_a_gap_layer()
    test_select_window_finds_true_contiguous_range()
    test_find_plateau_k_detects_plateau()
    test_find_plateau_k_uses_full_schedule_if_no_plateau()
    test_find_plateau_k_handles_trivial_inputs()
    print("All paper-fidelity fix tests passed (excluding the monkeypatch-based "
          "eta test, run via pytest).")
