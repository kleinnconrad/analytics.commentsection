"""Conservative vocabulary matching, not validated political stance inference."""

import re

import pandas as pd

from comment_analytics.config import Settings

LEXICON_VERSION = "exploratory-framing-v3"
# Explicit inflections avoid both substring matches and unrestricted stem matching.
# Ambiguous everyday terms (e.g. skilled workers, regulars' table) are not signals.
LEXICON = {
    "left-associated": (
        r"geflüchtete[nmrs]?",
        r"schutzsuchende[nmrs]?",
        r"menschen auf der flucht",
        r"asyl ist ein grundrecht",
        r"geistige brandstiftung",
    ),
    "right-associated": (
        r"bahnhofsklatscher(?:n|s)?",
        r"willkommensklatscher(?:n|s)?",
        r"asylflut",
        r"wirtschaftsflüchtling(?:e|en|s)?",
        r"asylforderer(?:n|s)?",
    ),
}
PATTERNS = {
    side: re.compile(
        r"\b(?:"
        + "|".join(f"(?P<term{index}>{term})" for index, term in enumerate(terms))
        + r")\b",
        re.IGNORECASE,
    )
    for side, terms in LEXICON.items()
}
AMBIGUOUS = re.compile(
    r'["„“”«»]|^\s*>|\b(?:nicht|kein(?:e|en|em|er|es)?|ohne)\b', re.IGNORECASE | re.MULTILINE
)
LABELS = ["left-associated", "right-associated", "mixed", "insufficient-evidence"]


def classify_vocabulary(text: str) -> str:
    """Return a vocabulary association, abstaining on empty, tied, or ambiguous text.

    Quotation and negation checks deliberately abstain rather than interpreting
    stance. They do not resolve sarcasm, indirect quotes, or contextual meaning.
    """
    if not isinstance(text, str) or not text.strip() or AMBIGUOUS.search(text):
        return "unknown"
    scores = {
        side: len({match.lastgroup for match in pattern.finditer(text)})
        for side, pattern in PATTERNS.items()
    }
    left, right = scores["left-associated"], scores["right-associated"]
    if left == right:
        return "unknown"
    return "left-associated" if left > right else "right-associated"


def aggregate_vocabulary(posts: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Retain every user and count the evidence behind each descriptive label.

    Args:
        posts: Enriched posts containing ID_User and text.
        settings: Minimum evidence and descriptive majority thresholds.

    Returns:
        One row per user including total, matched, and directional post counts.

    """
    posts = posts.assign(vocabulary=posts["text"].map(classify_vocabulary))
    counts = pd.crosstab(posts["ID_User"], posts["vocabulary"]).reindex(
        columns=["left-associated", "right-associated", "unknown"], fill_value=0
    )
    counts.columns = ["left_posts", "right_posts", "unknown_posts"]
    counts["matched_posts"] = counts["left_posts"] + counts["right_posts"]
    counts["total_posts"] = counts["matched_posts"] + counts["unknown_posts"]
    counts["match_coverage"] = counts["matched_posts"] / counts["total_posts"]
    denominator = counts["matched_posts"].where(counts["matched_posts"] > 0)
    counts["left_share"] = counts["left_posts"] / denominator
    counts["right_share"] = counts["right_posts"] / denominator
    enough = counts["matched_posts"] >= settings.min_political_posts
    counts["vocabulary_group"] = "insufficient-evidence"
    counts.loc[enough, "vocabulary_group"] = "mixed"
    for side in ("left", "right"):
        counts.loc[
            enough & (counts[f"{side}_share"] >= settings.political_majority), "vocabulary_group"
        ] = f"{side}-associated"
    return counts.reset_index()
