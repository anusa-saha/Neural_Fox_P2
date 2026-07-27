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

from .gpu_utils import free_memory


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


@torch.no_grad()
def capture_hidden_states_batched(
    model, tokenizer, model_cfg, layers: List[int], prompts: List[str],
    batch_size: int, device, store_device="cpu", min_batch_size: int = 1,
) -> Dict[int, torch.Tensor]:
    """Run `prompts` through the model in chunks of `batch_size`, capturing
    h^(l)(x) at the final token for every requested layer, WITHOUT ever
    building an autograd graph (`@torch.no_grad()` -- forgetting this is the
    single largest source of avoidable CUDA OOMs, since a full backward
    graph over an 8-9B model's forward pass costs far more memory than the
    forward activations themselves).

    On `torch.cuda.OutOfMemoryError`, the batch size is automatically halved
    (down to `min_batch_size`) and the *same* chunk of prompts is retried,
    so a single unlucky long prompt degrades gracefully instead of crashing
    the whole run.

    Returns {layer: Tensor[len(prompts), d_model]} on `store_device`
    (default "cpu" -- these tensors are tiny, batch x d_model, so moving them
    off the GPU costs nothing but keeps VRAM headroom for the next chunk).
    """
    all_chunks: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    i = 0
    bs = max(1, batch_size)
    while i < len(prompts):
        chunk = prompts[i:i + bs]
        try:
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            with ResidualCapture(model, model_cfg, layers) as cap:
                model(**enc)
                for l in layers:
                    all_chunks[l].append(cap.cache[l].to(store_device))
            del enc
            i += bs
        except torch.cuda.OutOfMemoryError:
            free_memory()
            if bs <= min_batch_size:
                raise
            bs = max(min_batch_size, bs // 2)

    return {l: torch.cat(chunks, dim=0) for l, chunks in all_chunks.items()}


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
