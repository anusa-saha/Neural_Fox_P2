"""
Neutral, weakly-specified, content-open prompts used for the defaultness
measurement and for the causal-lift calibration set (Stage I-B). These are
NOT translation sentences - they're exactly the kind of prompt the paper's
appendix describes for D_neutral: instruction-light, stratified by intent
class, and avoiding named entities / proper nouns / formatting templates
that could cue a language on their own (Appendix C, "Prompt sources,
suites, and splits").

All prompts are in English on purpose: the point is to see whether the
model's *default* continuation drifts to English even absent any English
instruction, and whether steering shifts that default towards the target
language.
"""

WEAK_PROMPTS = [
    # QA-like
    "What is the most interesting thing about",
    "Here is a question worth thinking about:",
    "Someone asked me an interesting question earlier.",
    "I wonder what the answer to this could be.",
    "Let me think about how to answer this properly.",
    # explanation-like
    "Let me explain how this works.",
    "Here's a simple way to think about it.",
    "To understand this, first consider the following.",
    "The reason this happens is not obvious at first.",
    "There are a few steps involved in explaining this clearly.",
    # summarization-like
    "Here is a short summary of what happened.",
    "In short, the main point was this.",
    "To sum up the discussion so far,",
    "The key takeaway from all of this is",
    "Overall, the situation can be described as follows.",
    # reasoning-like
    "Let's think through this step by step.",
    "First, consider what we already know.",
    "Given the information available, one could reason that",
    "It follows logically that",
    "Working through the possibilities one by one,",
    # dialogue-turn-like
    "That's a good point, and here's what I think.",
    "I see what you mean, let me respond to that.",
    "Thanks for sharing that, here is my reply.",
    "That reminds me of something worth mentioning.",
    "Building on what was just said,",
    # open continuation-like
    "It was a quiet afternoon when everything began.",
    "The old house had been empty for years.",
    "Nobody expected the meeting to go this way.",
    "The recipe called for a few simple ingredients.",
    "Walking down the street, one notices small details.",
    "The weather had been unpredictable all week.",
]
