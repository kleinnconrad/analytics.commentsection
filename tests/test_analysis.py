from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from comment_analytics.benchmark import benchmark, sentiment_scores
from comment_analytics.config import Settings
from comment_analytics.data import read_corpus
from comment_analytics.evaluation import staff_overlap
from comment_analytics.features import build_features, detect
from comment_analytics.political import aggregate_vocabulary, classify_vocabulary


def test_features_and_flags_preserve_missing_data_and_observable_behavior(data_dir: Path) -> None:
    posts, articles = read_corpus(data_dir)
    features, enriched = build_features(posts, articles)
    users, cutoff = detect(features, Settings())
    users = users.set_index("ID_User")
    assert users.index.tolist() == [1, 2, 3, 4, 5, 6]
    assert len(enriched) == 41
    assert enriched["ID_Post"].is_unique
    assert enriched.loc[enriched["ID_User"] == 1, "text"].eq("Den Geflüchteten helfen").all()
    assert users.loc[1, "duplicate_share"] == pytest.approx(0.9)
    assert users.loc[1, "flag_repetition"] == 1
    assert users.loc[2, "duplicate_share"] == 0
    assert users.loc[3, "text_posts"] == 0
    assert pd.isna(users.loc[3, "duplicate_share"])
    assert users.loc[3, "flag_repetition"] == 0
    # Five eligible users; the single-post high-vote user must not shift the cutoff.
    assert cutoff == pytest.approx(1.875)
    assert users.loc[6, "avg_controversy"] == pytest.approx(50 / 16)
    assert users["flag_any"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}


def test_literal_text_is_preserved_and_missing_topics_are_not_concentration(data_dir: Path) -> None:
    posts = pd.read_csv(data_dir / "Posts.csv", keep_default_na=False)
    posts.loc[:2, "Body"] = ["NA", "null", "001"]
    posts.to_csv(data_dir / "Posts.csv", index=False)
    articles = pd.read_csv(data_dir / "Articles.csv")
    articles["Path"] = ""
    articles.to_csv(data_dir / "Articles.csv", index=False)
    validated, articles = read_corpus(data_dir)
    assert validated.loc[:2, "Body"].tolist() == ["NA", "null", "001"]
    features, _ = build_features(validated, articles)
    users, _ = detect(features, Settings())
    assert users["topic_entropy"].isna().all()
    assert users["flag_topic"].sum() == 0


@pytest.mark.parametrize("score", [0, 3.25])
def test_impact_does_not_flag_quantile_ties(data_dir: Path, score: float) -> None:
    features, _ = build_features(*read_corpus(data_dir))
    features["avg_controversy"] = score
    users, cutoff = detect(features, Settings())
    assert cutoff == score
    assert users["flag_impact"].sum() == 0


def test_invalid_delays_do_not_become_immediate_responses(data_dir: Path) -> None:
    posts, articles = read_corpus(data_dir)
    posts["CreatedAt"] = pd.Timestamp("2015-06-01 09:00:00")
    articles["publishingDate"] = pd.Timestamp("2015-06-01 10:00:00")
    features, _ = build_features(posts, articles)
    users, _ = detect(features, Settings())
    assert users["article_delay_mean"].isna().all()
    assert users["avg_upvotes_per_article_delay"].isna().all()
    assert users["flag_temporal"].sum() == 0
    articles["publishingDate"] = pd.Timestamp("2015-06-01 08:00:00")
    features, _ = build_features(posts, articles)
    users, _ = detect(features, Settings())
    assert users.loc[users["ID_User"] == 1, "flag_temporal"].item() == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Stammtisch am Freitag", "unknown"),
        ("Wir brauchen Fachkräfte für den Arbeitsmarkt", "unknown"),
        ("Menschenrechte und Obergrenzen", "unknown"),
        ("Den Geflüchteten helfen", "left-associated"),
        ("Schutzsuchenden helfen", "left-associated"),
        ("Wirtschaftsflüchtling", "right-associated"),
        ("Wirtschaftsflüchtlinge", "right-associated"),
        ("Asylflutwarnung", "unknown"),
        ("Keine Asylflut", "unknown"),
        ('Die Zeitung schreibt: "Asylflut"', "unknown"),
        ("Geflüchtete Geflüchteten Asylflut", "unknown"),
        ("", "unknown"),
    ],
)
def test_vocabulary_associations_abstain_on_ambiguous_text(text: str, expected: str) -> None:
    assert classify_vocabulary(text) == expected


def test_political_minimum_and_inclusive_threshold_are_independent() -> None:
    rows = []
    for user, left, right, unknown in [(1, 1, 0, 50), (2, 17, 8, 2), (3, 16, 9, 0), (4, 0, 0, 5)]:
        for text, count in [("Geflüchtete", left), ("Asylflut", right), ("Other", unknown)]:
            rows.extend({"ID_User": user, "text": text} for _ in range(count))
    users = aggregate_vocabulary(pd.DataFrame(rows), Settings()).set_index("ID_User")
    assert users["vocabulary_group"].to_dict() == {
        1: "insufficient-evidence",
        2: "left-associated",
        3: "mixed",
        4: "insufficient-evidence",
    }
    assert users.loc[2, "left_share"] == pytest.approx(0.68)
    assert users.loc[2, "match_coverage"] == pytest.approx(25 / 27)
    assert pd.isna(users.loc[4, "left_share"])
    stricter = aggregate_vocabulary(pd.DataFrame(rows), replace(Settings(), min_political_posts=26))
    assert stricter["vocabulary_group"].eq("insufficient-evidence").all()


def test_staff_overlap_uses_users_and_keeps_undefined_rates_null(data_dir: Path) -> None:
    features, _ = build_features(*read_corpus(data_dir))
    users, _ = detect(features, Settings())
    result = staff_overlap(users, pd.read_csv(data_dir / "Newspaper_Staff.csv"))
    assert result["staff_without_posts"] == 1
    assert result["staff_in_posts"] == 2
    assert result["flag_any"] == {
        "flagged_staff": 1,
        "flagged_nonstaff": 1,
        "unflagged_staff": 1,
        "unflagged_nonstaff": 3,
        "staff_share_of_flagged": 0.5,
        "share_of_staff_flagged": 0.5,
    }
    assert result["flag_temporal"]["staff_share_of_flagged"] is None


@pytest.mark.parametrize(
    ("table", "column", "value", "message"),
    [
        ("Posts", "ID_Post", 2, "Duplicate ID_Post"),
        ("Articles", "ID_Article", 2, "Duplicate ID_Article"),
        ("Posts", "ID_User", None, "non-null integer"),
        ("Posts", "ID_Article", 900, "missing articles"),
        ("Posts", "CreatedAt", "bad date", "Unknown datetime|Unknown string"),
        ("Posts", "PositiveVotes", -1, "nonnegative integer"),
    ],
)
def test_invalid_corpus_is_rejected(
    data_dir: Path, table: str, column: str, value: object, message: str
) -> None:
    path = data_dir / f"{table}.csv"
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        read_corpus(data_dir)


def test_benchmark_metrics_are_not_micro_accuracy() -> None:
    result = sentiment_scores(
        ["negative", "negative", "neutral", "positive"],
        ["negative", "neutral", "neutral", "negative"],
    )
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    assert result["macro_f1"] == pytest.approx(7 / 18)


def test_benchmark_rejects_floating_model_revisions_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40_CHARACTER"):
        benchmark(tmp_path / "absent.csv", ["model@main"], "sentiment", tmp_path / "output")
