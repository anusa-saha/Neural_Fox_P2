"""
Build boolean vocab-sized masks V_target / V_english used to compute
early-step token-mass defaultness: M(x) = mass(V_target) - mass(V_english).

Simple heuristic (documented limitation, same one the paper itself flags
in its Limitations section - token-set construction is inherently imperfect):
  - Non-Latin-script targets (hi/bn/te/zh): a token counts as "target" if it
    contains a character in that language's Unicode block, "english" if it's
    plain ASCII alphabetic.
  - Spanish (shares Latin script with English): a token counts as "target" if
    it contains a Spanish-specific character (accents, ñ, ¿, ¡), "english" if
    it's plain ASCII alphabetic with no such characters.
"""
import re
import torch

UNICODE_RANGES = {
    "hi": [(0x0900, 0x097F)],   # Devanagari
    "bn": [(0x0980, 0x09FF)],   # Bengali
    "te": [(0x0C00, 0x0C7F)],   # Telugu
    "zh": [(0x4E00, 0x9FFF)],   # CJK Unified Ideographs
}
SPANISH_CHARS = set("áéíóúñüÁÉÍÓÚÑÜ¿¡")
ASCII_ALPHA_RE = re.compile(r"^[A-Za-z' \-]+$")


def _in_ranges(ch, ranges):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def build_token_masks(tokenizer, lang_key, device="cuda"):
    vocab_size = len(tokenizer)
    target_mask = torch.zeros(vocab_size, dtype=torch.bool)
    english_mask = torch.zeros(vocab_size, dtype=torch.bool)

    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        s = s.strip()
        if not s:
            continue
        if lang_key == "es":
            if any(c in SPANISH_CHARS for c in s):
                target_mask[tid] = True
            elif ASCII_ALPHA_RE.match(s):
                english_mask[tid] = True
        else:
            ranges = UNICODE_RANGES[lang_key]
            if any(_in_ranges(c, ranges) for c in s):
                target_mask[tid] = True
            elif ASCII_ALPHA_RE.match(s):
                english_mask[tid] = True

    return target_mask.to(device), english_mask.to(device)
