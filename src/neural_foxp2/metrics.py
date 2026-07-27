"""
Defaultness metrics (Sec. 2, "Language defaultness"; Appendix D.1).

    M_t^l(x)      = sum_{u in V_l} w(u) p_theta(u | ctx_t)
    Delta_M(x, t) = M_t^target(x) - M_t^english(x)

aggregated (mean) over the first T decoding steps, T in {1, 2, 3} by default
(the paper's early-horizon defaultness window).

Token-set construction (V_target, V_english) is a known soft spot flagged by
the paper itself (Appendix D.1.5, "token-set construction variants"). We use
a simple, documented heuristic:
  - For scripts with a distinct Unicode block (Hindi/Devanagari, Bengali,
    Telugu, Chinese/Han), V_target = tokens containing a character in that
    block; V_english = ASCII-alphabetic tokens.
  - For Spanish (shares Latin script with English), V_target = tokens
    containing a Spanish-specific diacritic/punctuation character
    (á é í ó ú ñ ¿ ¡); V_english = plain-ASCII-alphabetic tokens.
This is a coarse proxy, not a linguistically validated diagnostic set -- see
the paper's own discussion of shared-token inflation and transliteration
wins (Appendix D.1.5, Table 8) for why a production system should replace
this with a corpus-derived diagnostic + transliteration-aware token set.
"""
from typing import Sequence, Tuple
import torch

UNICODE_SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],  # Devanagari
    "bn": [(0x0980, 0x09FF)],  # Bengali
    "te": [(0x0C00, 0x0C7F)],  # Telugu
    "zh": [(0x4E00, 0x9FFF)],  # CJK Unified Ideographs
}
SPANISH_DIACRITIC_CHARS = set("áéíóúñÁÉÍÓÚÑ¿¡")


def _in_ranges(ch: str, ranges) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def build_token_sets(tokenizer, lang_code: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (target_token_ids, english_token_ids) as LongTensors, built
    directly in token-id space (Appendix D.1.1)."""
    vocab = tokenizer.get_vocab()
    tgt_ids, en_ids = [], []
    ranges = UNICODE_SCRIPT_RANGES.get(lang_code)

    for tok_str, tok_id in vocab.items():
        try:
            piece = tokenizer.convert_tokens_to_string([tok_str])
        except Exception:
            continue
        piece = piece.strip()
        if not piece:
            continue
        if ranges is not None:
            if any(_in_ranges(c, ranges) for c in piece):
                tgt_ids.append(tok_id)
            elif piece.isascii() and piece.isalpha():
                en_ids.append(tok_id)
        elif lang_code == "es":
            if any(c in SPANISH_DIACRITIC_CHARS for c in piece):
                tgt_ids.append(tok_id)
            elif piece.isascii() and piece.isalpha():
                en_ids.append(tok_id)
        else:
            raise ValueError(f"No token-set heuristic registered for lang_code={lang_code}")

    return (
        torch.tensor(sorted(set(tgt_ids)), dtype=torch.long),
        torch.tensor(sorted(set(en_ids)), dtype=torch.long),
    )


@torch.no_grad()
def next_token_distributions(model, tokenizer, prompts: Sequence[str], horizon: int, device) -> torch.Tensor:
    """Greedy-decode `horizon` steps, recording p_theta(. | ctx_t) at each step.

    Returns a tensor [n_prompts, horizon, vocab].
    """
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    all_probs = []
    past = None
    cur_ids = input_ids
    cur_attn = attn
    for _ in range(horizon):
        out = model(input_ids=cur_ids, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs)
        past = out.past_key_values
        next_ids = probs.argmax(dim=-1, keepdim=True)
        cur_ids = next_ids
        cur_attn = torch.cat([cur_attn, torch.ones_like(next_ids)], dim=-1)
    return torch.stack(all_probs, dim=1)  # [B, horizon, V]


def mass(probs_bt_v: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """M_t^l(x): (uniform-weight, w(u)=1 on-set) token-mass over V_l."""
    if token_ids.numel() == 0:
        return torch.zeros(probs_bt_v.shape[:-1], device=probs_bt_v.device)
    return probs_bt_v[..., token_ids].sum(dim=-1)  # [B, horizon]


def delta_m(probs_bt_v: torch.Tensor, tgt_ids: torch.Tensor, en_ids: torch.Tensor) -> torch.Tensor:
    """Delta_M(x, t) = M_t^target(x) - M_t^english(x), per prompt per step."""
    m_tgt = mass(probs_bt_v, tgt_ids)
    m_en = mass(probs_bt_v, en_ids)
    return m_tgt - m_en  # [B, horizon]; paper aggregates mean over t in {1,2,3}
