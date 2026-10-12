"""Render the political comparison charts and pages of results/prof_writing_analytics from a run."""

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comment_analytics.data import file_hash

LOG = logging.getLogger(__name__)
DIRECTIONS = {"left-associated": "#b84745", "right-associated": "#3973aa"}
GROUPS = {"Heuristic-flagged": "", "Unflagged": "normal_"}
YEARS = (2015, 2016)
TOP_TARGET_PATHS = 15
TOP_COMPARISON_PATHS = 10


def directional_posts(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load users and the posts of left- or right-associated users from a completed run.

    Args:
        run: Run directory written by ``comment-analytics run``.

    Returns:
        All users, and posts by directional users with year and path.

    """
    if not (run / "manifest.json").exists():
        raise ValueError(f"Run is incomplete or missing a manifest: {run}")
    users = pd.read_csv(run / "users.csv")
    posts = pd.read_csv(
        run / "predictions.csv",
        usecols=["ID_Post", "ID_User", "CreatedAt", "Path", "writer_group", "vocabulary_group"],
        keep_default_na=False,
        dtype={"Path": "string"},
    )
    posts = posts[posts["vocabulary_group"].isin(DIRECTIONS)].copy()
    posts["year"] = pd.to_datetime(posts["CreatedAt"], format="mixed").dt.year
    posts["Path"] = posts["Path"].replace("", "Unknown")
    return users, posts


def path_counts_chart(posts: pd.DataFrame, title: str, path: Path) -> bool:
    """Plot comment counts per path for left- and right-associated users.

    Returns:
        False when there are no posts, so no chart is written.

    """
    if posts.empty:
        return False
    top = posts["Path"].value_counts().head(TOP_TARGET_PATHS).index
    counts = (
        pd.crosstab(posts["Path"], posts["vocabulary_group"])
        .reindex(index=top, columns=list(DIRECTIONS), fill_value=0)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
    rows = np.arange(len(counts))
    for offset, (label, color) in zip((0.2, -0.2), DIRECTIONS.items(), strict=True):
        ax.barh(rows + offset, counts[label], height=0.4, color=color, label=label)
    ax.set_yticks(rows, counts.index)
    ax.set_xlabel("Comments by users with this vocabulary label (all their comments on the path)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def share_chart(users: pd.DataFrame, title: str, path: Path) -> pd.DataFrame:
    """Plot left/right shares among directional users per flag group and return the table."""
    table = (
        pd.crosstab(users["writer_group"], users["vocabulary_group"])
        .reindex(index=list(GROUPS), columns=list(DIRECTIONS), fill_value=0)
        .rename_axis(index=None, columns=None)
    )
    shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    rows = np.arange(len(GROUPS))
    for offset, (label, color) in zip((-0.2, 0.2), DIRECTIONS.items(), strict=True):
        bars = ax.bar(rows + offset, shares[label].fillna(0), width=0.4, color=color, label=label)
        for bar, value in zip(bars, shares[label], strict=True):
            if pd.notna(value):
                ax.annotate(
                    f"{value:.1f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                )
    ax.set_xticks(rows, [f"{group}\n(n={table.loc[group].sum():,})" for group in GROUPS])
    ax.set_ylabel("Share of left- and right-associated users (%)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def path_share_chart(posts: pd.DataFrame, title: str, path: Path) -> bool:
    """Plot left/right comment shares on the most commented paths, one panel per group."""
    if posts.empty:
        return False
    top = posts["Path"].value_counts().head(TOP_COMPARISON_PATHS).index
    fig, axes = plt.subplots(
        1, len(GROUPS), figsize=(18, 8), sharex=True, sharey=True, layout="constrained"
    )
    for ax, group in zip(axes, GROUPS, strict=True):
        members = posts[(posts["writer_group"] == group) & posts["Path"].isin(top)]
        counts = (
            pd.crosstab(members["Path"], members["vocabulary_group"])
            .reindex(index=top, columns=list(DIRECTIONS), fill_value=0)
            .iloc[::-1]
        )
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        rows = np.arange(len(shares))
        for offset, (label, color) in zip((0.2, -0.2), DIRECTIONS.items(), strict=True):
            bars = ax.barh(
                rows + offset, shares[label].fillna(0), height=0.4, color=color, label=label
            )
            for bar, value in zip(bars, shares[label], strict=True):
                if pd.notna(value):
                    ax.annotate(
                        f"{value:.0f}%",
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        va="center",
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=9,
                    )
        ax.set_yticks(rows, counts.index)
        ax.set_xlim(0, 110)
        ax.set_title(f"{group} (n={int(counts.to_numpy().sum()):,} comments)")
        ax.set_xlabel("Share of comments on the path (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def toc(*headings: str) -> str:
    """Return a Markdown table of contents for the given second-level headings."""
    anchors = (heading.lower().replace(" ", "-").replace(",", "") for heading in headings)
    lines = (f"- [{heading}](#{anchor})" for heading, anchor in zip(headings, anchors, strict=True))
    return "## Contents\n" + "\n".join(lines) + "\n"


def publish(run: Path, output: Path) -> None:
    """Overwrite the charts and pages in ``output`` with results of a completed run.

    Only the files named here are replaced; year charts without data are removed
    so that no stale image remains. Git history keeps earlier versions.

    Args:
        run: Completed run directory.
        output: Results directory, e.g. results/prof_writing_analytics.

    """
    users, posts = directional_posts(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    settings = manifest["configuration"]["settings"]
    output.mkdir(parents=True, exist_ok=True)
    majority = f"{settings['political_majority'] * 100:g}%"
    minimum = settings["min_political_posts"]
    rule = (
        f"at least {minimum} directional comment{'' if minimum == 1 else 's'} "
        f"and at least {majority} agreement"
    )

    table = share_chart(
        users,
        f"Left/right vocabulary among directional users\n({rule})",
        output / "ideological_share_comparison.png",
    )
    path_share_chart(
        posts,
        f"Left/right share of comments on the {TOP_COMPARISON_PATHS} most commented paths\n"
        f"(all comments of directional users; {rule})",
        output / "ideological_path_comparison.png",
    )
    timelines = {}
    for group, prefix in GROUPS.items():
        members = posts[posts["writer_group"] == group]
        sections = []
        for scope in ("total", *YEARS):
            name = f"political_spectrum_targets_{prefix}{scope}.png"
            subset = members if scope == "total" else members[members["year"] == scope]
            label = "all years" if scope == "total" else str(scope)
            if path_counts_chart(
                subset,
                f"Paths commented by directional {group.lower()} users ({label})",
                output / name,
            ):
                sections.append((label.capitalize() if scope == "total" else label, name))
            else:
                (output / name).unlink(missing_ok=True)
        timelines[group] = sections

    run_line = (
        f"Run `{run.name}`, manifest SHA-256 `{file_hash(run / 'manifest.json')[:12]}`, "
        f"Git revision `{(manifest.get('git_revision') or 'unknown')[:12]}`"
        f"{' (uncommitted changes)' if manifest.get('working_tree_dirty') else ''}, "
        f"lexicon `{manifest['configuration']['lexicon_version']}`."
    )
    flagged = int((users["writer_group"] == "Heuristic-flagged").sum())
    directional = int(users["vocabulary_group"].isin(DIRECTIONS).sum())
    rows = "\n".join(
        f"| {group} | {int((users['writer_group'] == group).sum()):,} | "
        f"{table.loc[group, 'left-associated']:,} | {table.loc[group, 'right-associated']:,} | "
        + " | ".join(
            "n/a"
            if table.loc[group].sum() == 0
            else f"{table.loc[group, label] / table.loc[group].sum():.1%}"
            for label in DIRECTIONS
        )
        + " |"
        for group in GROUPS
    )
    (output / "README.md").write_text(
        f"""# Political vocabulary comparison

{toc("Source", "Method", "Results", "Limitations", "Files")}
## Source

{run_line}
The run contains {manifest["counts"]["posts"]:,} posts and {manifest["counts"]["users"]:,} users.
Regenerate these files with `comment-analytics publish --run <run directory> --output results/prof_writing_analytics`.

## Method

- **Comment label:** the comment body is matched against the lexicon `{manifest["configuration"]["lexicon_version"]}`
  (the "crisis" lexicon of the former main branch). The side with more distinct matched terms labels the comment.
- **User label:** {rule}. Users below the agreement threshold are mixed; users without directional comments are not labelled.
- **Groups:** "Heuristic-flagged" users meet at least one behavioral flag (temporal, topic, similarity, impact);
  all other users are "Unflagged". See the [root README](../../README.md#methodology).

## Results

{directional:,} of {manifest["counts"]["users"]:,} users are left- or right-associated; {flagged:,} users are flagged.

| Flag group | Users | Left-associated | Right-associated | Left share | Right share |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

Shares use only left- and right-associated users as the denominator.

## Limitations

- With one directional comment, a single word decides a user's label.
- Generic terms such as `menschenrechte`, `menschlichkeit`, `stammtisch`, and `obergrenze` produce most matches.
- Negation, quotation, and sarcasm are not interpreted.
- Flags describe behavior. They are not evidence of employment, coordination, or astroturfing.
- Path charts count all comments of directional users on a path, including nonpolitical comments.

## Files

- [Group comparison](political_comparison_report.md)
- [Flagged users by path and year](political_timeline.md)
- [Unflagged users by path and year](political_timeline_normal.md)
""",
        encoding="utf-8",
    )
    (output / "political_comparison_report.md").write_text(
        f"""# Political vocabulary: flagged and unflagged users

{toc("Share of left and right users", "Share by path")}
{run_line}

## Share of left and right users
Left- and right-associated users within each flag group ({rule}).

![Share comparison](ideological_share_comparison.png)

## Share by path
Left/right share of all comments by directional users on the {TOP_COMPARISON_PATHS} paths with the most such comments.

![Path comparison](ideological_path_comparison.png)
""",
        encoding="utf-8",
    )
    for group, filename in (
        ("Heuristic-flagged", "political_timeline.md"),
        ("Unflagged", "political_timeline_normal.md"),
    ):
        count = int(table.loc[group].sum())
        sections = timelines[group]
        body = "\n".join(f"## {label}\n![{label}]({name})\n" for label, name in sections)
        (output / filename).write_text(
            f"""# Paths commented by directional {group.lower()} users

{toc(*(label for label, _ in sections)) if sections else ""}
{run_line}

Directional users: **{count:,}** ({rule}). Each chart shows the {TOP_TARGET_PATHS} paths with
the most comments by these users.

{body}""",
            encoding="utf-8",
        )
    LOG.info("Published %s directional users from %s to %s", directional, run, output)
