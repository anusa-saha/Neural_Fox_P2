"""
Build matched English <-> target-language sentence pairs from
openlanguagedata/flores_plus. FLORES is sentence-aligned across languages
via a shared 'id' column, which is exactly the "meaning unit" pairing
Stage I needs (same content, different language).
"""
from datasets import load_dataset


def load_matched_pairs(lang_flores_code, n_sentences, english_code, splits=("dev", "devtest")):
    en_rows, tgt_rows = {}, {}
    for split in splits:
        en_ds = load_dataset("openlanguagedata/flores_plus", english_code, split=split)
        tgt_ds = load_dataset("openlanguagedata/flores_plus", lang_flores_code, split=split)
        for row in en_ds:
            en_rows[row["id"]] = row["text"]
        for row in tgt_ds:
            tgt_rows[row["id"]] = row["text"]

    common_ids = sorted(set(en_rows) & set(tgt_rows))[:n_sentences]
    pairs = [(en_rows[i], tgt_rows[i]) for i in common_ids]
    if len(pairs) < n_sentences:
        print(f"[data_utils] only found {len(pairs)} matched pairs for {lang_flores_code} "
              f"(requested {n_sentences})")
    return pairs
