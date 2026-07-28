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

    Unlike the old (deprecated) `facebook/flores` dataset, `openlanguagedata/flores_plus`
    does NOT expose hyphenated "eng_Latn-hin_Deva" pair configs or `sentence_<lang>`
    columns. Each language is its own single-language config (e.g. "hin_Deva") with a
    flat `text` column, and rows are aligned across languages via a shared `id` column
    (scoped to `split`). So we load the English config and the target config
    separately and join on id.

    Note: flores_plus is a gated dataset. You must (1) accept the terms on
    https://huggingface.co/datasets/openlanguagedata/flores_plus while logged in, and
    (2) be authenticated locally (`huggingface-cli login`, or pass a token via the
    HF_TOKEN env var), or `load_dataset` will raise an auth/403 error here regardless
    of the config name being correct.
    """
    from datasets import load_dataset  # local import: only needed for real runs

    if lang_code not in LANGUAGES:
        raise ValueError(f"Unknown language code {lang_code}; add it to config.LANGUAGES")
    flores_tgt = LANGUAGES[lang_code]["flores"]

    ds_en = load_dataset("openlanguagedata/flores_plus", ENGLISH_FLORES, split=split)
    ds_tgt = load_dataset("openlanguagedata/flores_plus", flores_tgt, split=split)

    # Row order isn't guaranteed to match across single-language configs, so join
    # explicitly on the shared sentence id rather than assuming aligned indices.
    en_by_id = {row["id"]: row["text"] for row in ds_en}
    tgt_by_id = {row["id"]: row["text"] for row in ds_tgt}
    common_ids = sorted(set(en_by_id) & set(tgt_by_id))
    if not common_ids:
        raise ValueError(
            f"No overlapping sentence ids between {ENGLISH_FLORES!r} and {flores_tgt!r} "
            f"for split={split!r}; check that both configs actually cover this split."
        )

    rng = random.Random(seed)
    rng.shuffle(common_ids)
    chosen = common_ids[: min(n, len(common_ids))]

    return [
        MeaningUnit(uid=f"{flores_tgt}-{split}-{sid}", text_en=en_by_id[sid], text_tgt=tgt_by_id[sid])
        for sid in chosen
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