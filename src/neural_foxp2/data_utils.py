"""
Matched multilingual prompt construction for Neural FOXP2.

Implements the "meaning unit" pairing described in the paper (Sec. 2.1,
"Activation collection and SAE training"; Appendix C, "Meaning units and
matched pairing for Delta_Z"): for a fixed semantic item we build two prompt
realizations, cond=en and cond=lt (target language), holding everything but
language identity fixed, so that

    Delta_z(x) = z(x_lt) - z(x_en)

isolates language identity from content. We source translation-equivalent
sentences from FLORES+ (dev/devtest splits), which is exactly the corpus
used for the translation evaluation in Table 2.
"""
from dataclasses import dataclass
from typing import List, Tuple
import random

from .config import LANGUAGES, ENGLISH_FLORES


@dataclass
class MeaningUnit:
    uid: str
    text_en: str
    text_tgt: str


# Weak / neutral prompt templates used to probe *early* language defaultness
# under weak prompting (Sec. 2, "English is the lingua franca"). These do NOT
# specify a language, so any target-language mass shift reflects the model's
# internal generation prior (plus our steering), not instruction-following.
WEAK_PROMPT_TEMPLATES = [
    "{text}\n\nContinue:",
    "{text}\n\nWhat happens next?",
    "Here is a passage:\n{text}\n\nSummarize it.",
    "{text}\n\nAnswer the following question about the above passage.",
    "Rewrite the following in your own words:\n{text}",
]


def load_flores_pairs(lang_code: str, split: str = "devtest", n: int = 200, seed: int = 0) -> List[MeaningUnit]:
    """Load n translation-matched (English, target) sentence pairs from FLORES+.

    `lang_code` must be a key of config.LANGUAGES (e.g. "hi", "es", "zh", ...).
    Uses the HF `facebook/flores` dataset, config name "{eng_flores}-{tgt_flores}"
    (e.g. "eng_Latn-hin_Deva"), which is sentence-aligned by row index.
    """
    from datasets import load_dataset  # local import: only needed for real runs

    if lang_code not in LANGUAGES:
        raise ValueError(f"Unknown language code {lang_code}; add it to config.LANGUAGES")
    flores_tgt = LANGUAGES[lang_code]["flores"]
    config_name = f"{ENGLISH_FLORES}-{flores_tgt}"

    ds = load_dataset("facebook/flores", config_name, split=split, trust_remote_code=True)
    col_en = f"sentence_{ENGLISH_FLORES}"
    col_tgt = f"sentence_{flores_tgt}"

    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]

    return [
        MeaningUnit(uid=f"{flores_tgt}-{split}-{i}", text_en=ds[i][col_en], text_tgt=ds[i][col_tgt])
        for i in idx
    ]


def build_matched_prompts(pairs: List[MeaningUnit], template: str = "{text}") -> Tuple[List[str], List[str]]:
    """Wrap meaning-unit sentences into matched (English, target) prompts.

    Both conditions use the *same* template; surface form differs only in
    which language's sentence fills the slot (Sec. 2.1).
    """
    en_prompts = [template.format(text=p.text_en) for p in pairs]
    tgt_prompts = [template.format(text=p.text_tgt) for p in pairs]
    return en_prompts, tgt_prompts


def build_weak_prompts(pairs: List[MeaningUnit], rng_seed: int = 0) -> List[str]:
    """Weak/neutral prompts (D_weak) used for the causal-lift probe (Sec.
    2.1.2), the window-selection gain probe (Sec. 2.2), and the English
    attractor estimate mu_en (Sec. 2.3(b)).

    These do not specify a language and use only the English sentence as
    content, so any target-language mass increase reflects a shift in the
    model's *default* generation language rather than instruction following.
    """
    rng = random.Random(rng_seed)
    out = []
    for p in pairs:
        t = rng.choice(WEAK_PROMPT_TEMPLATES)
        out.append(t.format(text=p.text_en))
    return out
