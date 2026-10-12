"""Read and validate corpus data without silently dropping or multiplying posts."""

import hashlib
import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

POST_COLUMNS = {
    "ID_Post",
    "ID_Article",
    "ID_User",
    "CreatedAt",
    "Headline",
    "Body",
    "PositiveVotes",
    "NegativeVotes",
}
ARTICLE_COLUMNS = {"ID_Article", "publishingDate", "Path"}


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    """Raise a descriptive error when a table lacks required columns."""
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def validate_ids(frame: pd.DataFrame, columns: list[str], unique: str) -> None:
    """Validate non-null integer identifiers and the table's unique primary key."""
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or ((values % 1) != 0).any():
            raise ValueError(f"{column} must contain non-null integer identifiers")
        frame[column] = values.astype("int64")
    if frame[unique].duplicated().any():
        raise ValueError(f"Duplicate {unique} identifiers")


def read_corpus(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV exports and reject invalid keys, votes, timestamps, or article references.

    Args:
        data_dir: Directory containing Posts.csv and Articles.csv.

    Returns:
        Validated posts and articles with parsed timestamps.

    """
    posts = pd.read_csv(
        data_dir / "Posts.csv",
        keep_default_na=False,
        dtype={"Headline": "string", "Body": "string"},
    )
    articles = pd.read_csv(
        data_dir / "Articles.csv", keep_default_na=False, dtype={"Path": "string"}
    )
    require_columns(posts, POST_COLUMNS, "Posts")
    require_columns(articles, ARTICLE_COLUMNS, "Articles")
    if posts.empty or articles.empty:
        raise ValueError("Posts and Articles must not be empty")
    validate_ids(posts, ["ID_Post", "ID_Article", "ID_User"], "ID_Post")
    validate_ids(articles, ["ID_Article"], "ID_Article")
    if not posts["ID_Article"].isin(articles["ID_Article"]).all():
        raise ValueError("Posts reference missing articles")
    for frame, column in ((posts, "CreatedAt"), (articles, "publishingDate")):
        frame[column] = pd.to_datetime(frame[column], errors="raise", format="mixed")
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing timestamps")
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            raise ValueError(f"{column} has inconsistent timestamp timezones")
    if posts["CreatedAt"].dt.tz != articles["publishingDate"].dt.tz:
        raise ValueError("Post and article timestamp timezones must agree")
    for column in ("PositiveVotes", "NegativeVotes"):
        values = pd.to_numeric(posts[column], errors="raise")
        if values.isna().any() or (values < 0).any() or ((values % 1) != 0).any():
            raise ValueError(f"{column} must contain nonnegative integer vote counts")
        posts[column] = values
    return posts, articles


def combine_text(posts: pd.DataFrame) -> pd.Series:
    """Combine headline and body; retain empty text as empty rather than invented content."""
    return (
        posts["Headline"].fillna("").astype(str).str.strip()
        + "\n"
        + posts["Body"].fillna("").astype(str).str.strip()
    ).str.strip()


def export_database(database: Path, output: Path) -> None:
    """Export existing SQLite tables in chunks using a read-only connection.

    Args:
        database: Existing corpus database, never created implicitly.
        output: Destination directory for CSV tables.

    """
    database = database.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Export directory is not empty: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if not tables:
            raise ValueError("Database contains no exportable tables")
        for (table,) in tables:
            if not table.replace("_", "").isalnum():
                raise ValueError(f"Unsupported table name: {table!r}")
            destination = output / f"{table}.csv"
            with destination.open("w", encoding="utf-8", newline="") as stream:
                first = True
                for chunk in pd.read_sql_query(
                    f'SELECT * FROM "{table}"', connection, chunksize=50_000
                ):
                    chunk.to_csv(stream, index=False, header=first)
                    first = False
    finally:
        connection.close()


def file_hash(path: Path) -> str:
    """Return a streaming SHA-256 digest for an input or output file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(inputs: list[Path], configuration: dict[str, Any]) -> dict[str, Any]:
    """Describe exact inputs, installed dependencies, source revision, and configuration."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty = None, None
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "working_tree_dirty": dirty,
        "source_sha256": {path.name: file_hash(path) for path in source_files},
        "inputs": {str(path.resolve()): file_hash(path) for path in inputs},
        "configuration": configuration,
        "packages": {
            name: version(name)
            for name in (
                "comment-section-analytics",
                "pandas",
                "numpy",
                "scipy",
                "matplotlib",
                "gensim",
            )
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write strict UTF-8 JSON, rejecting nonfinite values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
