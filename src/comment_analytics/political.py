"""Lexicon matching from the main branch, not validated political stance inference."""

import re

import pandas as pd

from comment_analytics.config import Settings

LEXICON_VERSION = "main-crisis-v1"
# The "crisis" lexicon of the former src/config/political_config.py, unchanged.
# Terms match lowercase text as whole words in the listed form only; inflections
# (e.g. "geflüchteten") do not match. Generic terms such as "menschenrechte" or
# "obergrenze" are used by commenters of every orientation.
LEXICON = {
    "left-associated": (
        "schutzsuchende",
        "menschen auf der flucht",
        "geflüchtete",
        "menschlichkeit",
        "menschenrechte",
        "asyl ist ein grundrecht",
        "hassposter",
        "geistige brandstiftung",
        "stammtisch",
    ),
    "right-associated": (
        "bahnhofsklatscher",
        "willkommensklatscher",
        "asylflut",
        "wirtschaftsflüchtlinge",
        "asylforderer",
        "kontrollverlust",
        "obergrenze",
        "fachkräfte",
        "goldstücke",
        "kulturbereicherer",
    ),
}
PATTERNS = {
    side: re.compile(
        r"\b(?:"
        + "|".join(f"(?P<term{index}>{re.escape(term)})" for index, term in enumerate(terms))
        + r")\b"
    )
    for side, terms in LEXICON.items()
}
# Body text only, as on the main branch; headlines are not classified.
TEXT_COLUMN = "Body"
LABELS = ["left-associated", "right-associated", "mixed", "insufficient-evidence"]


def classify_vocabulary(text: str) -> str:
    """Return the side with more distinct matched terms, or unknown on ties and no match.

    Negation, quotation, sarcasm, and context are not interpreted.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    lower = text.lower()
    scores = {
        side: len({match.lastgroup for match in pattern.finditer(lower)})
        for side, pattern in PATTERNS.items()
    }
    left, right = scores["left-associated"], scores["right-associated"]
    if left == right:
        return "unknown"
    return "left-associated" if left > right else "right-associated"


def aggregate_vocabulary(posts: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Retain every user and count the evidence behind each descriptive label.

    Args:
        posts: Enriched posts containing ID_User and Body.
        settings: Minimum evidence and descriptive majority thresholds.

    Returns:
        One row per user including total, matched, and directional post counts.

    """
    posts = posts.assign(vocabulary=posts[TEXT_COLUMN].map(classify_vocabulary))
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
