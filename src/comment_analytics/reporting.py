"""Render descriptive reports whose denominators include unclassified users."""

from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comment_analytics.config import Settings
from comment_analytics.political import LABELS, LEXICON_VERSION

DIRECTIONS = ["left-associated", "right-associated"]
GROUPS = ["Unflagged", "Heuristic-flagged"]


def write_reports(
    users: pd.DataFrame, predictions: pd.DataFrame, output: Path, settings: Settings
) -> None:
    """Save chart data, a user-share plot, path counts, sensitivity, and a Markdown report.

    User shares use every user in each flag group, including mixed and insufficient
    evidence. Path shares count all posts by labelled users, not political posts.
    No confidence intervals are claimed for unvalidated labels or a nonrandom sample.
    """
    counts = pd.crosstab(users["writer_group"], users["vocabulary_group"]).reindex(
        index=GROUPS, columns=LABELS, fill_value=0
    )
    totals = counts.sum(axis=1)
    shares = counts.div(totals.replace(0, np.nan), axis=0)
    table = counts.rename_axis(index="writer_group", columns="vocabulary_group").stack()
    table = table.rename("users").reset_index()
    table["group_users"] = table["writer_group"].map(totals)
    table["share"] = table["users"] / table["group_users"].replace(0, np.nan)
    # Share among left- and right-associated users only, as charted on the main branch.
    directional = counts[DIRECTIONS].sum(axis=1)
    table["directional_users"] = table["writer_group"].map(directional)
    table["directional_share"] = (
        table["users"] / table["directional_users"].replace(0, np.nan)
    ).where(table["vocabulary_group"].isin(DIRECTIONS))
    table.to_csv(output / "user_shares.csv", index=False)
    evidence = (
        f"At least {settings.min_political_posts} matched "
        f"comment{'' if settings.min_political_posts == 1 else 's'}"
    )

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    left = np.zeros(len(GROUPS))
    colors = ["#b84745", "#3973aa", "#b99135", "#b9bdc5"]
    for label, color in zip(LABELS, colors, strict=True):
        widths = shares[label].fillna(0).to_numpy() * 100
        ax.barh(GROUPS, widths, left=left, label=label, color=color)
        for row, width in enumerate(widths):
            if width >= 8:
                ax.text(left[row] + width / 2, row, f"{width:.1f}%", ha="center", va="center")
        left += widths
    ax.set_yticks(range(len(GROUPS)), [f"{group}\n(n={totals[group]:,})" for group in GROUPS])
    for row, group in enumerate(GROUPS):
        if totals[group] == 0:
            ax.text(2, row, "No users; share unavailable", va="center")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all users within each flag group (%)")
    ax.set_title(
        "Exploratory vocabulary associations, not inferred ideology\n"
        f"{evidence} and at least {settings.political_majority * 100:g}% directional agreement"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2, frameon=False)
    fig.savefig(output / "vocabulary_shares.png", dpi=160)
    plt.close(fig)

    path_posts = predictions.assign(
        year=predictions["CreatedAt"].dt.year,
        Path=predictions["Path"].replace("", pd.NA).fillna("Unknown"),
    )
    path_counts = (
        path_posts.groupby(["year", "Path", "writer_group", "vocabulary_group"])
        .agg(comments=("ID_Post", "size"), users=("ID_User", "nunique"))
        .reset_index()
    )
    path_counts["path_group_comments"] = path_counts.groupby(["year", "Path", "writer_group"])[
        "comments"
    ].transform("sum")
    path_counts["comment_share"] = path_counts["comments"] / path_counts["path_group_comments"]
    path_counts.to_csv(output / "path_comment_shares.csv", index=False)

    sensitivity = []
    for minimum in sorted({1, 3, 5, 10, 20, settings.min_political_posts}):
        for majority in sorted({0.6, 0.68, 0.8, settings.political_majority}):
            # Reuse saved directional counts; never reclassify text for each threshold.
            scenario = replace(settings, min_political_posts=minimum, political_majority=majority)
            classified = (users["matched_posts"] >= scenario.min_political_posts) & (
                users[["left_share", "right_share"]].max(axis=1) >= scenario.political_majority
            )
            for group in GROUPS:
                members = users["writer_group"] == group
                sensitivity.append(
                    {
                        "min_matched_posts": minimum,
                        "majority": majority,
                        "writer_group": group,
                        "users": int(members.sum()),
                        "directional_users": int((members & classified).sum()),
                    }
                )
    pd.DataFrame(sensitivity).to_csv(output / "threshold_sensitivity.csv", index=False)
    first, last = predictions["CreatedAt"].min(), predictions["CreatedAt"].max()
    settings_table = "\n".join(f"| {key} | {value} |" for key, value in asdict(settings).items())
    group_rows = []
    for group in GROUPS:
        members = users[users["writer_group"] == group]
        group_rows.append(
            f"| {group} | {len(members):,} | {int(members['total_posts'].sum()):,} | "
            f"{int(members['matched_posts'].sum()):,} |"
        )
    direction_rows = [
        f"| {group} | {counts.loc[group, DIRECTIONS[0]]:,} | {counts.loc[group, DIRECTIONS[1]]:,} | "
        + " | ".join(
            "n/a"
            if directional[group] == 0
            else f"{counts.loc[group, label] / directional[group]:.1%}"
            for label in DIRECTIONS
        )
        + " |"
        for group in GROUPS
    ]
    report = f"""# Exploratory comment analysis

## Contents
- [Scope](#scope)
- [User vocabulary shares](#user-vocabulary-shares)
- [Outputs and limitations](#outputs-and-limitations)
- [Configuration](#configuration)

## Scope
This run contains {len(predictions):,} posts and {len(users):,} users.
Observed post timestamps: {first.isoformat()} to {last.isoformat()}.
These extrema are not evidence of continuous collection coverage.
Business hours use timestamps as supplied; no unverified timezone conversion is applied.

| Flag group | Users | All comments | Directionally matched comments |
| --- | ---: | ---: | ---: |
{chr(10).join(group_rows)}

## User vocabulary shares
Lexicon: `{LEXICON_VERSION}`, applied to comment bodies. Directional labels require:
{evidence.lower()} and at least {settings.political_majority * 100:g}% agreement.
These are descriptive thresholds, not significance tests. With one matched comment, a single
word decides a user's label. Unknown comments are excluded from directional agreement, but their
users remain in chart denominators.

![Vocabulary shares](vocabulary_shares.png)

Among left- and right-associated users only (the comparison charted on the main branch):

| Flag group | Left-associated | Right-associated | Left share | Right share |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(direction_rows)}

## Outputs and limitations
- [User shares](user_shares.csv): counts and denominators for every label, including insufficient evidence.
- [Path shares](path_comment_shares.csv): activity-weighted counts of all comments by user label,
  grouped by observed year and path. They are not counts of ideological statements or evidence of targeting.
  Absent rows mean zero observed comments; no share is estimated for an unobserved path/group.
- [Threshold sensitivity](threshold_sensitivity.csv): directional user counts across several cutoffs.
- [Staff overlap](staff_overlap.json): overlap only; nonstaff users are not verified negative examples.
- [Lexicon candidates](lexicon_candidates.csv): Word2Vec nearest neighbours of lexicon terms.
  They share contexts, often the topic rather than the stance, and require manual review before use.
- [Word vectors](word2vec.bin): the gensim Word2Vec vectors trained on this corpus (binary word2vec format).
- [Manifest](manifest.json): input and output hashes, source identity, dependencies, and configuration.

Flags describe behavior, not employment, coordination, deception, or astroturfing.
The lexicon is not validated for stance, sarcasm, or ideological identity. Quote and negation
abstention is conservative but incomplete. Selection and activity differences confound group comparisons.
There is no held-out political accuracy estimate and no justified population confidence interval.
No pretrained model is used. Word2Vec is trained on the analysed comments, so similarity reflects
this corpus only. High comment similarity can come from copied text, templates, or a narrow topic;
it is not proof of copy-pasted propaganda.

## Configuration
| Setting | Value |
| --- | ---: |
{settings_table}

Similarity and impact use the eligible-user quantile and strictly greater scores; cutoff ties
are excluded. Similarity is the mean pairwise cosine similarity of a user's comment vectors
(averaged unit word vectors with the corpus mean removed), rounded to six decimals.
Missing measurements are unavailable, not zero. Negative publication delays are excluded,
and affected users cannot satisfy the temporal flag.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
