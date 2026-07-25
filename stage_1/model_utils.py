"""
Batched versions of everything GPU-bound in Stage I. Every function here
chunks its input to `max_batch_size` internally, so callers just pass full
lists of texts (or full candidate x calibration cross-products) and get
per-row results back - no manual chunking needed at the call site.

Left-padding + explicit position_ids is used throughout so that (a) the
final *real* token is always at sequence position -1 regardless of padding,
and (b) RoPE-based models (Llama/Qwen/Gemma all use RoPE) get correct
position indices for the padded rows instead of counting pad tokens.
"""
import torch


def prepare_tokenizer_for_batching(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_layer_module(model, hook_module_path, layer):
    mod = model
    for name in hook_module_path.format(layer=layer).split("."):
        mod = getattr(mod, name)
    return mod


def _left_pad_batch(tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    attn = enc["attention_mask"]
    position_ids = attn.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attn == 0, 0)
    return enc["input_ids"], attn, position_ids


def token_mass_batch(logits, mask):
    """logits: [batch, vocab] -> returns [batch] mass under `mask`."""
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[:, mask].sum(dim=-1)


def get_last_token_activations_batch(model, tokenizer, texts, layer_module, device, max_batch_size):
    """Returns a [len(texts), hidden] tensor of the residual-stream activation
    at each text's final real token (single forward pass per chunk, no generation).

    Calls model.model(...) (the transformer backbone) rather than model(...)
    (the full CausalLM wrapper) - we only need the hooked layer's output, not
    logits, and skipping the LM head avoids materializing a [batch, seq_len,
    vocab_size] tensor for models with huge vocabularies (e.g. Gemma-2's 256k),
    which is otherwise the single biggest memory cost in this function.
    """
    backbone = getattr(model, "model", model)
    all_acts = []
    for start in range(0, len(texts), max_batch_size):
        chunk = texts[start:start + max_batch_size]
        input_ids, attn, position_ids = _left_pad_batch(tokenizer, chunk, device)

        store = {}
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store["h"] = hs[:, -1, :].detach()
        handle = layer_module.register_forward_hook(hook)
        with torch.no_grad():
            backbone(input_ids=input_ids, attention_mask=attn, position_ids=position_ids)
        handle.remove()

        all_acts.append(store["h"].cpu())
    return torch.cat(all_acts, dim=0)


def _forward_last_logits(model, input_ids, attn, position_ids):
    """Runs the full CausalLM forward but asks it to only materialize logits
    for the final sequence position (we only ever read out.logits[:, -1, :]
    anyway). Without this, HF's default forward computes logits for EVERY
    position, i.e. a [batch, seq_len, vocab_size] tensor - for a big vocab
    (e.g. Gemma-2's 256k) and a long sequence (non-Latin scripts often
    tokenize much less efficiently, so "long" can sneak up on you), this is
    frequently the single largest tensor in the whole forward pass and the
    most common source of OOM in this pipeline. logits_to_keep=1 cuts that
    down to [batch, 1, vocab_size]. Tries both the current and the older
    parameter name, then falls back to a plain call if neither is supported.
    """
    try:
        return model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                      logits_to_keep=1)
    except TypeError:
        pass
    try:
        return model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                      num_logits_to_keep=1)
    except TypeError:
        pass
    return model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids)


def _horizon_chunk(model, tokenizer, texts, device, horizon, target_mask, english_mask,
                    layer_module, delta_batch):
    """One chunk's worth of greedy decoding (batch <= max_batch_size), with an
    optional per-row residual-stream delta added at the last position on
    every decoding step. delta_batch: None, or [batch, hidden] tensor."""
    input_ids, attn, position_ids = _left_pad_batch(tokenizer, texts, device)
    prompt_len = input_ids.shape[1]

    handle = None
    if layer_module is not None and delta_batch is not None:
        def hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] = hs[:, -1, :] + delta_batch.to(hs.dtype)
                return (hs,) + tuple(out[1:])
            else:
                out[:, -1, :] = out[:, -1, :] + delta_batch.to(out.dtype)
                return out
        handle = layer_module.register_forward_hook(hook)

    batch = input_ids.shape[0]
    per_step_mass = [[] for _ in range(batch)]
    with torch.no_grad():
        for _ in range(horizon):
            out = _forward_last_logits(model, input_ids, attn, position_ids)
            logits = out.logits[:, -1, :]
            mass_t = token_mass_batch(logits, target_mask)
            mass_e = token_mass_batch(logits, english_mask)
            diff = (mass_t - mass_e).tolist()
            for i, d in enumerate(diff):
                per_step_mass[i].append(d)

            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_ids], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_ids)], dim=1)
            position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)

    if handle is not None:
        handle.remove()

    mean_mass = [sum(m) / len(m) for m in per_step_mass]
    continuations = tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
    return mean_mass, continuations


def batched_horizon_defaultness(model, tokenizer, texts, device, horizon, target_mask, english_mask,
                                 layer_module=None, delta_vecs=None, max_batch_size=256):
    """
    texts: list[str], length N. Can be a plain list of weak prompts, or a
    tiled "candidates x calibration" cross-product - the caller controls that.

    delta_vecs:
      - None                     -> no intervention (plain greedy decoding)
      - 1-D tensor [hidden]      -> the SAME edit applied to every row
      - 2-D tensor [N, hidden]   -> a DIFFERENT edit per row (e.g. one row
                                     per (candidate feature, calibration prompt) pair)

    Returns (mean_mass_per_row: list[float] len N, continuations: list[str] len N),
    aggregating Delta_M over t=1..horizon for each row.
    """
    n = len(texts)
    mean_mass_all = [None] * n
    continuations_all = [None] * n

    for start in range(0, n, max_batch_size):
        end = min(start + max_batch_size, n)
        chunk_texts = texts[start:end]

        chunk_delta = None
        if delta_vecs is not None:
            chunk_delta = delta_vecs if delta_vecs.dim() == 1 else delta_vecs[start:end]
            if chunk_delta.dim() == 1:
                chunk_delta = chunk_delta.unsqueeze(0).expand(len(chunk_texts), -1)

        mass_list, conts = _horizon_chunk(
            model, tokenizer, chunk_texts, device, horizon, target_mask, english_mask,
            layer_module, chunk_delta,
        )
        mean_mass_all[start:end] = mass_list
        continuations_all[start:end] = conts

    return mean_mass_all, continuations_all
