"""
Loaders for the three pretrained SAE suites. All of them get wrapped into
the same tiny SimpleSAE interface (encode / decoder_row) so the rest of the
pipeline doesn't care which family it's using.
"""
import os
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file


class SimpleSAE:
    """z = relu(x @ W_enc + b_enc);  dh/dz_j = W_dec[j]  (decoder row for feature j)."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec

    def to(self, device):
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)
        return self

    def encode(self, x):
        return torch.relu(x.float() @ self.W_enc.float() + self.b_enc.float())

    def decoder_row(self, feature_idx):
        return self.W_dec[feature_idx].float()


def _wrap_saelens(sae):
    return SimpleSAE(
        W_enc=sae.W_enc.data.clone(),
        b_enc=sae.b_enc.data.clone(),
        W_dec=sae.W_dec.data.clone(),
        b_dec=sae.b_dec.data.clone(),
    )


def _fix_orientation(W_enc, b_enc, W_dec, b_dec):
    """Normalize to W_enc: [d_model, d_sae], W_dec: [d_sae, d_model], using the
    bias vector lengths (d_sae for b_enc, d_model for b_dec) as ground truth -
    more robust than guessing from the weight shapes alone."""
    d_sae = b_enc.numel()
    d_model = b_dec.numel()
    if W_enc.shape[1] != d_sae:
        W_enc = W_enc.T.contiguous()
    if W_dec.shape[1] != d_model:
        W_dec = W_dec.T.contiguous()
    return W_enc, b_enc, W_dec, b_dec


def _pick(sd, *names):
    for n in names:
        if n in sd:
            return sd[n]
    raise KeyError(f"None of {names} found in checkpoint keys: {list(sd.keys())[:20]}")


def _flatten_checkpoint(obj):
    """Some .pt checkpoints wrap the actual tensors under 'state_dict'/'model'/'sae'."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "sae"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_gemma_scope_sae(layer, width="16k", variant="it", device="cuda"):
    from sae_lens import SAE
    release = f"gemma-scope-9b-{variant}-res-canonical"
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, _cfg, _sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return _wrap_saelens(sae).to(device)


def load_llama_scope_sae(repo_id, subfolder, device="cuda"):
    """Llama Scope: per-layer subfolder containing a .safetensors checkpoint."""
    try:
        from sae_lens import SAE
        sae, _cfg, _sparsity = SAE.from_pretrained(release=repo_id, sae_id=subfolder, device=device)
        return _wrap_saelens(sae).to(device)
    except Exception as e:
        print(f"[sae_utils] sae_lens failed for {repo_id}/{subfolder} ({e}); "
              f"falling back to raw safetensors loading.")

    local_dir = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
    folder = os.path.join(local_dir, subfolder)
    weight_file = next((os.path.join(folder, f) for f in os.listdir(folder)
                         if f.endswith(".safetensors")), None)
    if weight_file is None:
        raise FileNotFoundError(f"No .safetensors file found in {folder}")
    sd = load_file(weight_file)

    W_enc = _pick(sd, "W_enc", "encoder.weight")
    b_enc = _pick(sd, "b_enc", "encoder.bias")
    W_dec = _pick(sd, "W_dec", "decoder.weight")
    b_dec = _pick(sd, "b_dec", "decoder.bias", "b_dec_out")
    W_enc, b_enc, W_dec, b_dec = _fix_orientation(W_enc, b_enc, W_dec, b_dec)
    return SimpleSAE(W_enc, b_enc, W_dec, b_dec).to(device)


def load_qwen_scope_sae(repo_id, layer, filename_template, layer_index_base, device="cuda"):
    """Qwen SAE-Res: a single .pt file per layer directly in the repo root,
    e.g. layer01.pt, layer02.pt, ... (confirmed convention)."""
    file_layer_num = layer + layer_index_base
    filename = filename_template.format(layer=file_layer_num)
    path = hf_hub_download(repo_id, filename)
    obj = torch.load(path, map_location="cpu")
    sd = _flatten_checkpoint(obj)

    W_enc = _pick(sd, "W_enc", "encoder.weight")
    b_enc = _pick(sd, "b_enc", "encoder.bias")
    W_dec = _pick(sd, "W_dec", "decoder.weight")
    b_dec = _pick(sd, "b_dec", "decoder.bias", "b_dec_out")
    W_enc, b_enc, W_dec, b_dec = _fix_orientation(W_enc, b_enc, W_dec, b_dec)
    return SimpleSAE(W_enc, b_enc, W_dec, b_dec).to(device)


def get_sae_for_layer(model_key, model_cfg, layer, device="cuda"):
    family = model_cfg["family"]
    if family == "gemma":
        return load_gemma_scope_sae(
            layer, width=model_cfg["gemma_width"], variant=model_cfg["gemma_variant"], device=device
        )
    elif family == "llama":
        subfolder = model_cfg["sae_subfolder_template"].format(layer=layer)
        return load_llama_scope_sae(model_cfg["sae_repo"], subfolder, device=device)
    elif family == "qwen":
        return load_qwen_scope_sae(
            model_cfg["sae_repo"], layer,
            model_cfg["sae_filename_template"], model_cfg["sae_layer_index_base"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
