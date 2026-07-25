"""
Script-agnostic LID validator channel (Appendix D.2 in the paper).

This is deliberately kept separate from token_sets.py: the token-mass
channel (V_target / V_english) has to classify individual *vocabulary
tokens*, where a real LID model doesn't work well (most subword tokens
are too short / ambiguous out of context). The LID channel instead runs
on the *decoded generated continuation text*, where a real LID model is
reliable, and serves as a cross-check that the mass channel's gains
aren't just token-set artifacts (Appendix D.2.2 "cross-metric validity").

Uses py3langid (pure-python port of langid.py, no external model download).
"""
import py3langid as langid
from py3langid.langid import LanguageIdentifier, MODEL_FILE


def build_lid_identifier(candidate_langs):
    """candidate_langs: list of ISO-639-1 codes, e.g. ['en', 'hi'].
    MODEL_FILE is a *path* to the pickled default model, so it must be loaded
    with from_pickled_model - from_modelstring expects the raw base64 model
    string itself, not a path, and will fail with a base64 padding error."""
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    identifier.set_languages(candidate_langs)
    return identifier


def lid_target_minus_english(identifier, text, target_lang_code, english_code="en"):
    """Returns P(target | text) - P(english | text) under the restricted
    (target, english) LID distribution. 0.0 for empty/degenerate text."""
    text = (text or "").strip()
    if not text:
        return 0.0
    ranked = identifier.rank(text)  # list of (lang, prob), normalized over candidate set
    probs = dict(ranked)
    return probs.get(target_lang_code, 0.0) - probs.get(english_code, 0.0)
