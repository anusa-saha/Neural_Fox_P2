"""
GPU memory configuration and reporting utilities.

Tuned so the full Neural FOXP2 pipeline (base model + per-layer pretrained
SAEs + Stage I/II/III activation probing) fits comfortably on a single
RTX PRO 6000 (Blackwell workstation, 96 GB VRAM) without a CUDA OOM, while
still batching everything so the exact same code scales *down* gracefully on
smaller GPUs instead of crashing.

Three independent levers keep memory bounded:
  1. Every forward pass used only for activation capture / scoring / metric
     probing runs under `torch.no_grad()` (see activations.py, metrics.py) --
     omitting this was the single largest OOM risk in the original code,
     since it silently built a full autograd graph on an 8-9B model for
     every Stage I/II probe.
  2. Prompt lists are always chunked to `GPUBudget.prompt_batch_size` /
     `lift_probe_batch_size` rather than forwarded in one unbounded batch,
     with automatic batch-size halving on `torch.cuda.OutOfMemoryError`
     (`safe_batched_call`).
  3. Per-layer SAEs are loaded in bf16 by default (halving their footprint
     vs. the fp32 checkpoints) and non-window-layer SAEs are freed from GPU
     memory once Stage II selects the final intervention window, since only
     those layers' SAEs are needed for Stage III generation-time steering.
"""
import gc
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch


# RTX PRO 6000 (Blackwell workstation) ships with 96 GB of VRAM. These
# defaults leave several GB of headroom for the CUDA context, allocator
# fragmentation, and the transient KV-cache spikes from Stage I's
# causal-lift decoding loop (many short forward passes in quick succession).
RTX_PRO_6000_VRAM_GB = 96


@dataclass
class GPUBudget:
    prompt_batch_size: int = 16        # matched/weak prompts per activation-capture forward pass
    lift_probe_batch_size: int = 32    # weak prompts per causal-lift / gain-probe decode
    generate_batch_size: int = 8       # prompts per batched .generate() call
    sae_dtype: torch.dtype = torch.bfloat16
    offload_non_window_saes: bool = True
    empty_cache_between_stages: bool = True
    min_batch_size: int = 1            # floor for automatic OOM backoff


def detect_vram_gb(device: str = "cuda") -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    idx = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(idx)
    return props.total_memory / (1024 ** 3)


def recommended_budget(device: str = "cuda") -> GPUBudget:
    """Scale batch sizes to whatever VRAM is actually present. The top tier
    (>=80 GB) targets RTX PRO 6000 / A100-80GB / H100-class cards; smaller
    GPUs get smaller batches automatically so the same code runs slower --
    not OOM -- on e.g. a 24 GB card."""
    vram = detect_vram_gb(device)
    if vram is None:
        return GPUBudget()
    if vram >= 80:
        return GPUBudget(prompt_batch_size=32, lift_probe_batch_size=64, generate_batch_size=16)
    if vram >= 40:
        return GPUBudget(prompt_batch_size=16, lift_probe_batch_size=32, generate_batch_size=8)
    if vram >= 20:
        return GPUBudget(prompt_batch_size=8, lift_probe_batch_size=16, generate_batch_size=4)
    return GPUBudget(prompt_batch_size=4, lift_probe_batch_size=8, generate_batch_size=2)


def free_memory():
    """Release Python-side references and hand fragmented CUDA blocks back
    to the allocator. Cheap; call liberally at stage/job boundaries."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_snapshot(device: str = "cuda") -> dict:
    """A JSON-friendly snapshot of current GPU memory usage, for logging
    into run_config.json / memory_report.json."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    idx = torch.device(device).index or 0
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_properties(idx).name,
        "total_gb": round(torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1024 ** 3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1024 ** 3, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(idx) / 1024 ** 3, 2),
    }


def safe_batched_call(
    items: List[Any], fn: Callable[[List[Any]], Any], batch_size: int,
    min_batch_size: int = 1, combine: str = "cat",
):
    """Apply `fn` to `items` in chunks of `batch_size`, automatically halving
    the batch size (down to `min_batch_size`) and retrying on
    `torch.cuda.OutOfMemoryError` before giving up (re-raising once the
    floor is hit). This is the core "don't crash, just slow down" mechanism
    used by every batched forward pass in the pipeline.

    `combine`:
      - "cat":  fn returns a Tensor per chunk; results are torch.cat'd on dim 0.
      - "list": fn returns a list per chunk; results are concatenated as a list.
      - "raw":  return the raw list of per-chunk outputs, uncombined.
    """
    outputs = []
    i = 0
    bs = max(1, batch_size)
    while i < len(items):
        chunk = items[i:i + bs]
        try:
            out = fn(chunk)
            outputs.append(out)
            i += bs
        except torch.cuda.OutOfMemoryError:
            free_memory()
            if bs <= min_batch_size:
                raise
            bs = max(min_batch_size, bs // 2)

    if combine == "cat":
        return torch.cat(outputs, dim=0)
    if combine == "list":
        flat = []
        for o in outputs:
            flat.extend(o)
        return flat
    return outputs
