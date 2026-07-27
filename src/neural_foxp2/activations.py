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
