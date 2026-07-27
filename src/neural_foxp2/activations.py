"""
Residual-stream hook utilities shared across all three Neural FOXP2 stages.

Every model family in config.MODELS exposes decoder layers at
`hook_module_path.format(layer=l)` (e.g. "model.layers.{layer}" for
Llama/Gemma/Qwen-style architectures). We hook the *output* of that module,
which is the residual-stream activation h^(l)(x) used throughout the paper,
e.g. Sec. 2.1: z^(l)(x) = ReLU(W_l^T h^(l)(x) + b_l).

No model-family-specific code lives here -- everything is driven by
config.MODELS[...]["hook_module_path"], so adding a new model family only
requires a config.py entry + a sae_utils.py loader.
"""
from typing import Callable, Dict, List
import torch


@torch.no_grad()
def capture_hidden_states(
    model, model_cfg, tokenizer, prompts: List[str], layers: List[int], device,
    batch_size: int = 16, token_pos: str = "last",
) -> Dict[int, torch.Tensor]:
    """Run `prompts` through `model` and capture h^(l)(x) at `layers`, for all
    prompts, as one concatenated tensor per layer (order-preserving).

    This is the memory-safe way to do activation capture for Stage I/II/III,
    for two reasons:

    - `torch.no_grad()`: nothing in this pipeline ever calls `.backward()` --
      capture is inference-only -- but a bare `model(**enc)` call still builds
      and retains a full autograd graph by default, which keeps *every*
      intermediate activation in *every* decoder layer alive (in an 8B model,
      that's dominated by the MLP up/gate/down-proj intermediates, each
      hidden_size -> ~3.5x hidden_size). For a few hundred prompts at once
      that graph is easily tens of GB larger than the handful of captured
      per-layer vectors we actually keep. This is almost always the real
      cause of a CUDA OOM here, well before model size or GPU choice matters.
    - `batch_size` chunking: even under no_grad, a single forward pass over
      *all* prompts at once pads every sequence to the longest prompt in the
      whole set and holds one full batch's worth of hidden states per layer
      simultaneously. Chunking bounds peak memory to one batch's worth,
      independent of how large n_disc/n_calib/n_weak is.
    """
    out: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        with ResidualCapture(model, model_cfg, layers, token_pos=token_pos) as cap:
            _ = model(**enc)
            for l in layers:
                out[l].append(cap.cache[l])
        del enc
    return {l: torch.cat(v, dim=0) for l, v in out.items()}


def get_module(model, path: str):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def layer_module(model, model_cfg, layer: int):
    path = model_cfg["hook_module_path"].format(layer=layer)
    return get_module(model, path)


class ResidualCapture:
    """Captures h^(l)(x) at the final prompt token for a set of layers.

    Used by Stage I/II to extract activations for matched (English, target)
    prompt pairs and weak/neutral prompts (Sec. 2.1, "Activation collection").
    """

    def __init__(self, model, model_cfg, layers: List[int], token_pos: str = "last"):
        self.model = model
        self.model_cfg = model_cfg
        self.layers = layers
        self.token_pos = token_pos
        self.cache: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.token_pos == "last":
                self.cache[layer] = h[:, -1, :].detach()
            else:
                self.cache[layer] = h.detach()
        return hook

    def __enter__(self):
        for l in self.layers:
            mod = layer_module(self.model, self.model_cfg, l)
            self._handles.append(mod.register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


class ResidualSteer:
    """Adds an additive vector to h^(l)(x) at every forward pass, at one or
    more layers simultaneously.

    `layer_fns[l]` is a callable `h -> delta_h` (same shape as h, broadcastable
    over batch/sequence). Stage I's causal-lift micro-intervention uses a
    constant `delta_h = alpha * W_dec[j]`; Stage III's suppression term uses a
    state-dependent function of the current hidden state.
    """

    def __init__(self, model, model_cfg, layer_fns: Dict[int, Callable[[torch.Tensor], torch.Tensor]]):
        self.model = model
        self.model_cfg = model_cfg
        self.layer_fns = layer_fns
        self._handles = []

    def _make_hook(self, layer):
        fn = self.layer_fns[layer]

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            delta = fn(h)
            h = h + delta
            if is_tuple:
                return (h,) + output[1:]
            return h

        return hook

    def __enter__(self):
        for l in self.layer_fns:
            mod = layer_module(self.model, self.model_cfg, l)
            self._handles.append(mod.register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
