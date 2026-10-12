"""Compute observable user behavior without inferring motives or professional status."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import entropy

from comment_analytics.config import FLAGS, Settings
from comment_analytics.data import combine_text

LOG = logging.getLogger(__name__)


def build_features(
    posts: pd.DataFrame, articles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute user features and retain an identifiable row for each validated post.

    Args:
        posts: Validated posts with datetime timestamps.
        articles: Validated articles with unique IDs.

    Returns:
        User features and enriched posts. Unmeasured quantities remain NaN.

    """
    merged = posts.merge(
        articles[["ID_Article", "publishingDate", "Path"]], on="ID_Article", validate="many_to_one"
    )
    merged["text"] = combine_text(merged)
    merged["topic"] = (
        merged["Path"]
        .fillna("")
        .map(lambda path: str(path).split("/")[1] if "/" in str(path) else str(path))
        .replace("", pd.NA)
    )
    created = merged["CreatedAt"]
    merged["business_hours"] = (created.dt.dayofweek < 5) & created.dt.hour.between(9, 17)
    delay = (created - merged["publishingDate"]).dt.total_seconds() / 60
    if (delay < 0).any():
        LOG.warning("Excluding %s negative article-to-comment delays", int((delay < 0).sum()))
    merged["article_delay_minutes"] = delay.where(delay >= 0)
    positive = merged["PositiveVotes"].astype(float)
    negative = merged["NegativeVotes"].astype(float)
    merged["controversy"] = positive * negative / (positive + negative + 1)
    # Descriptive only: this is NOT a voting rate and does not participate in flags.
    merged["upvotes_per_article_delay"] = positive / (merged["article_delay_minutes"] + 1)
    merged["has_text"] = merged["text"].str.len() > 0
    users = merged.groupby("ID_User").agg(
        total_posts=("ID_Post", "size"),
        text_posts=("has_text", "sum"),
        business_hours_share=("business_hours", "mean"),
        article_delay_mean=("article_delay_minutes", "mean"),
        valid_delay_posts=("article_delay_minutes", "count"),
        avg_controversy=("controversy", "mean"),
        avg_upvotes_per_article_delay=("upvotes_per_article_delay", "mean"),
        active_days=("CreatedAt", lambda values: values.dt.date.nunique()),
        topic_posts=("topic", "count"),
        topic_entropy=(
            "topic",
            lambda values: entropy(values.value_counts()) if values.notna().any() else np.nan,
        ),
    )
    users["posts_per_active_day"] = users["total_posts"] / users["active_days"]
    text_posts = merged.loc[merged["has_text"], ["ID_User", "text"]].copy()
    text_posts["normalized"] = (
        text_posts["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    )
    # Repeated occurrences after the first, divided by nonempty posts. No semantic claims.
    unique_texts = text_posts.groupby("ID_User")["normalized"].nunique()
    users["duplicate_share"] = 1 - unique_texts / users["text_posts"].replace(0, np.nan)
    return users.reset_index(), merged


def detect(users: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, float | None]:
    """Apply descriptive rules and return the eligible-user impact cutoff.

    Impact requires a score strictly above the quantile, so ties at the cutoff
    are excluded and an all-zero distribution flags nobody. Missing features
    never satisfy a rule. The four rules are combined with OR, not a probability.
    """
    users = users.copy()
    eligible = users["total_posts"] >= settings.min_posts
    users["flag_temporal"] = (
        (users["posts_per_active_day"] >= settings.temporal_posts_per_day)
        & (users["business_hours_share"] >= settings.business_hours_share)
        & (users["article_delay_mean"] <= settings.response_minutes)
        & (users["valid_delay_posts"] == users["total_posts"])
    )
    users["flag_topic"] = (
        (users["total_posts"] >= settings.min_topic_posts)
        & (users["topic_posts"] == users["total_posts"])
        & (users["topic_entropy"] <= settings.max_topic_entropy)
    )
    users["flag_repetition"] = (users["text_posts"] >= settings.min_posts) & (
        users["duplicate_share"] >= settings.duplicate_share
    )
    cutoff = users.loc[eligible, "avg_controversy"].quantile(settings.impact_quantile)
    users["flag_impact"] = (users["avg_controversy"] > cutoff) & (users["avg_controversy"] > 0)
    for flag in FLAGS:
        users[flag] = (users[flag] & eligible).astype(int)
    users["flag_any"] = users[FLAGS].max(axis=1)
    return users, None if pd.isna(cutoff) else float(cutoff)
