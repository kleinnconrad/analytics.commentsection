import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from comment_analytics.data import file_hash


def run_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("comment-analytics")
    assert executable is not None, "Install the package before running tests"
    return subprocess.run(
        [executable, *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONWARNINGS": "error"},
        check=False,
    )


@pytest.mark.parametrize("empty_group", [False, True])
def test_sqlite_to_report_from_unrelated_working_directory(
    tmp_path: Path, corpus: tuple[pd.DataFrame, ...], empty_group: bool
) -> None:
    database = tmp_path / "corpus.sqlite3"
    posts, articles, staff = corpus
    if empty_group:
        posts = posts[posts["ID_User"] == 3]
        staff = staff.iloc[:0]
    with closing(sqlite3.connect(database)) as connection:
        for name, frame in [("Posts", posts), ("Articles", articles), ("Newspaper_Staff", staff)]:
            frame.to_sql(name, connection, index=False)
    database_hash = file_hash(database)
    data = tmp_path / "export"
    output = tmp_path / "run"
    cwd = tmp_path / "unrelated"
    cwd.mkdir()
    exported = run_cli(cwd, "export", "--database", str(database), "--output", str(data))
    assert exported.returncode == 0, exported.stderr
    assert file_hash(database) == database_hash
    result = run_cli(cwd, "run", "--data-dir", str(data), "--output", str(output))
    assert result.returncode == 0, result.stderr
    predictions = pd.read_csv(output / "predictions.csv")
    assert set(predictions["ID_Post"]) == set(posts["ID_Post"])
    assert len(predictions) == len(posts)
    shares = pd.read_csv(output / "user_shares.csv")
    unflagged = shares[shares["writer_group"] == "Unflagged"].set_index("vocabulary_group")
    if empty_group:
        assert unflagged.loc["insufficient-evidence", "share"] == 1
        assert shares.loc[shares["writer_group"] == "Heuristic-flagged", "share"].isna().all()
    else:
        # User 1 writes only headlines and is flagged; bodies decide labels.
        assert unflagged["share"].to_dict() == {
            "left-associated": 0.25,
            "right-associated": 0.25,
            "mixed": 0.25,
            "insufficient-evidence": 0.25,
        }
        assert unflagged.loc["left-associated", "directional_share"] == 0.5
        assert pd.isna(unflagged.loc["mixed", "directional_share"])
        assert unflagged.loc["mixed", "users"] == 1
    paths = pd.read_csv(output / "path_comment_shares.csv")
    assert paths["comments"].sum() == len(posts)
    totals = paths.groupby(["year", "Path", "writer_group"])["comment_share"].sum()
    assert totals.tolist() == pytest.approx([1] * len(totals))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"posts": len(posts), "users": posts["ID_User"].nunique()}
    assert manifest["configuration"]["models"] == []
    assert manifest["embeddings_deterministic"] is True
    assert set(manifest["cutoffs"]) == {"similarity", "impact"}
    users = pd.read_csv(output / "users.csv")
    assert {"semantic_similarity", "similarity_posts", "flag_similarity"} <= set(users.columns)
    # No fixture word reaches the default minimum count of 20, so no candidates are proposed.
    candidates = pd.read_csv(output / "lexicon_candidates.csv")
    assert candidates.empty
    assert "margin" in candidates.columns
    if empty_group:
        # Only empty comments: no vocabulary, so no model and no similarity.
        assert not (output / "word2vec.bin").exists()
        assert users["semantic_similarity"].isna().all()
    else:
        assert (output / "word2vec.bin").stat().st_size > 0
    for name, digest in manifest["outputs"].items():
        assert file_hash(output / name) == digest
    assert manifest["inputs"][str(data / "Posts.csv")] == file_hash(data / "Posts.csv")
    assert (output / "vocabulary_shares.png").read_bytes().startswith(b"\x89PNG")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "at least 1 matched comment and at least 68% agreement" in report
    if not empty_group:
        assert "| Unflagged | 1 | 1 | 50.0% | 50.0% |" in report
    published = tmp_path / "published"
    result = run_cli(cwd, "publish", "--run", str(output), "--output", str(published))
    assert result.returncode == 0, result.stderr
    readme = (published / "README.md").read_text(encoding="utf-8")
    assert "at least 1 directional comment and at least 68% agreement" in readme
    timeline = (published / "political_timeline_normal.md").read_text(encoding="utf-8")
    if empty_group:
        assert "| Unflagged | 1 | 0 | 0 | n/a | n/a |" in readme
        assert not list(published.glob("political_spectrum_targets_*.png"))
        assert "Directional users: **0**" in timeline
    else:
        assert "| Unflagged | 4 | 1 | 1 | 50.0% | 50.0% |" in readme
        # The fixture only has 2015 posts; no 2016 chart is written.
        assert (published / "political_spectrum_targets_normal_2015.png").exists()
        assert not (published / "political_spectrum_targets_normal_2016.png").exists()
        assert "![2015](political_spectrum_targets_normal_2015.png)" in timeline
    assert (published / "ideological_share_comparison.png").read_bytes().startswith(b"\x89PNG")
    refused = run_cli(cwd, "run", "--data-dir", str(data), "--output", str(output))
    assert refused.returncode != 0
    assert "not empty" in refused.stderr
    refused = run_cli(cwd, "export", "--database", str(database), "--output", str(data))
    assert refused.returncode != 0
    assert "not empty" in refused.stderr


def test_missing_database_fails_without_creating_it(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite3"
    result = run_cli(
        tmp_path, "export", "--database", str(database), "--output", str(tmp_path / "csv")
    )
    assert result.returncode != 0
    assert not database.exists()
