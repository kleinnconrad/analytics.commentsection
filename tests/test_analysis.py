from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comment_analytics.benchmark import benchmark, sentiment_scores
from comment_analytics.config import Settings
from comment_analytics.data import read_corpus
from comment_analytics.embeddings import (
    comment_vectors,
    lexicon_candidates,
    train_embeddings,
    user_similarity,
)
from comment_analytics.evaluation import staff_overlap
from comment_analytics.features import build_features, detect
from comment_analytics.political import PATTERNS, aggregate_vocabulary, classify_vocabulary

SMALL = Settings(
    embedding_min_count=1, embedding_dimensions=8, embedding_epochs=3, lexicon_min_count=1
)


def analyse(
    posts: pd.DataFrame, articles: pd.DataFrame, settings: Settings = SMALL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, enriched = build_features(posts, articles)
    similarity = user_similarity(enriched, train_embeddings(enriched["text"], settings))
    return features.merge(similarity, on="ID_User", validate="one_to_one"), enriched


def test_features_and_flags_preserve_missing_data_and_observable_behavior(data_dir: Path) -> None:
    posts, articles = read_corpus(data_dir)
    features, enriched = analyse(posts, articles)
    users, cutoffs = detect(features, SMALL)
    users = users.set_index("ID_User")
    assert users.index.tolist() == [1, 2, 3, 4, 5, 6]
    assert len(enriched) == 41
    assert enriched["ID_Post"].is_unique
    assert enriched.loc[enriched["ID_User"] == 1, "text"].eq("Den Geflüchteten helfen").all()
    assert users.loc[1, "duplicate_share"] == pytest.approx(0.9)
    # Digits are not tokens, so "Ordinary text 1" and "Ordinary text 2" are identical.
    assert users.loc[[1, 2, 6], "semantic_similarity"].eq(1).all()
    assert users.loc[5, "semantic_similarity"] < 1
    # Three of four measured users tie at the cutoff, so nobody is flagged.
    assert cutoffs["similarity"] == 1
    assert users["flag_similarity"].sum() == 0
    assert users.loc[3, "text_posts"] == 0
    assert users.loc[3, "similarity_posts"] == 0
    assert pd.isna(users.loc[3, "semantic_similarity"])
    assert users.loc[3, "flag_similarity"] == 0
    # Five eligible users; the single-post high-vote user must not shift the cutoff.
    assert cutoffs["impact"] == pytest.approx(1.875)
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
    features, _ = analyse(validated, articles)
    users, _ = detect(features, SMALL)
    assert users["topic_entropy"].isna().all()
    assert users["flag_topic"].sum() == 0


@pytest.mark.parametrize(
    ("column", "name", "flag"),
    [
        ("avg_controversy", "impact", "flag_impact"),
        ("semantic_similarity", "similarity", "flag_similarity"),
    ],
)
@pytest.mark.parametrize("score", [0, 0.75])
def test_quantile_flags_do_not_flag_ties(
    data_dir: Path, column: str, name: str, flag: str, score: float
) -> None:
    features, _ = analyse(*read_corpus(data_dir))
    features[column] = score
    users, cutoffs = detect(features, SMALL)
    assert cutoffs[name] == score
    assert users[flag].sum() == 0


def test_similarity_flags_scores_above_the_eligible_quantile(data_dir: Path) -> None:
    features, _ = analyse(*read_corpus(data_dir))
    features["semantic_similarity"] = [0.1, 0.2, 0.3, 0.99, 0.5, 0.6]
    features["similarity_posts"] = 10
    users, cutoffs = detect(features, SMALL)
    # User 4 has one post and is ineligible, so 0.99 does not raise the cutoff.
    assert cutoffs["similarity"] == pytest.approx(0.596)
    assert users.set_index("ID_User")["flag_similarity"].to_dict() == {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
    }


def test_invalid_delays_do_not_become_immediate_responses(data_dir: Path) -> None:
    posts, articles = read_corpus(data_dir)
    posts["CreatedAt"] = pd.Timestamp("2015-06-01 09:00:00")
    articles["publishingDate"] = pd.Timestamp("2015-06-01 10:00:00")
    features, _ = analyse(posts, articles)
    users, _ = detect(features, SMALL)
    assert users["article_delay_mean"].isna().all()
    assert users["avg_upvotes_per_article_delay"].isna().all()
    assert users["flag_temporal"].sum() == 0
    articles["publishingDate"] = pd.Timestamp("2015-06-01 08:00:00")
    features, _ = analyse(posts, articles)
    users, _ = detect(features, SMALL)
    assert users.loc[users["ID_User"] == 1, "flag_temporal"].item() == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Generic terms of the main-branch lexicon count as signals.
        ("Stammtisch am Freitag", "left-associated"),
        ("Wir brauchen Fachkräfte für den Arbeitsmarkt", "right-associated"),
        ("Menschenrechte und Obergrenzen", "left-associated"),
        # Only the listed forms match; inflections and compounds do not.
        ("Den Geflüchteten helfen", "unknown"),
        ("Schutzsuchenden helfen", "unknown"),
        ("Wirtschaftsflüchtling", "unknown"),
        ("WIRTSCHAFTSFLÜCHTLINGE", "right-associated"),
        ("Asylflutwarnung", "unknown"),
        # Negation and quotation are not interpreted.
        ("Keine Asylflut", "right-associated"),
        ('Die Zeitung schreibt: "Asylflut"', "right-associated"),
        # Distinct terms per side are counted; repeats count once, ties are unknown.
        ("Geflüchtete Geflüchtete Asylflut", "unknown"),
        ("Menschenrechte, Menschlichkeit und Asylflut", "left-associated"),
        ("Menschen auf der Flucht", "left-associated"),
        ("", "unknown"),
    ],
)
def test_vocabulary_matches_main_branch_rules(text: str, expected: str) -> None:
    assert classify_vocabulary(text) == expected


def test_political_minimum_and_inclusive_threshold_are_independent() -> None:
    rows = []
    for user, left, right, unknown in [(1, 1, 0, 50), (2, 17, 8, 2), (3, 16, 9, 0), (4, 0, 0, 5)]:
        for text, count in [("Geflüchtete", left), ("Asylflut", right), ("Other", unknown)]:
            rows.extend({"ID_User": user, "Body": text} for _ in range(count))
    # One matched comment is enough by default, as on the main branch.
    default = aggregate_vocabulary(pd.DataFrame(rows), Settings()).set_index("ID_User")
    assert default.loc[1, "vocabulary_group"] == "left-associated"
    users = aggregate_vocabulary(
        pd.DataFrame(rows), replace(Settings(), min_political_posts=10)
    ).set_index("ID_User")
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


def test_headlines_are_not_classified() -> None:
    frame = pd.DataFrame(
        {"ID_User": [1, 2], "Headline": ["Asylflut", ""], "Body": [pd.NA, "Asylflut"]}
    )
    users = aggregate_vocabulary(frame, Settings()).set_index("ID_User")
    assert users["vocabulary_group"].to_dict() == {
        1: "insufficient-evidence",
        2: "right-associated",
    }


def test_staff_overlap_uses_users_and_keeps_undefined_rates_null(data_dir: Path) -> None:
    features, _ = analyse(*read_corpus(data_dir))
    users, _ = detect(features, SMALL)
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


def test_word2vec_training_is_reproducible_with_one_worker(data_dir: Path) -> None:
    _, enriched = build_features(*read_corpus(data_dir))
    first = train_embeddings(enriched["text"], SMALL)
    second = train_embeddings(enriched["text"], SMALL)
    assert first is not None and second is not None
    assert first.index_to_key == second.index_to_key
    assert (first.vectors == second.vectors).all()


def test_user_similarity_equals_mean_pairwise_cosine(data_dir: Path) -> None:
    _, enriched = build_features(*read_corpus(data_dir))
    vectors = train_embeddings(enriched["text"], SMALL)
    users = user_similarity(enriched, vectors).set_index("ID_User")
    matrix = comment_vectors(enriched["text"], vectors)
    rows = matrix[(enriched["ID_User"] == 5).to_numpy()]
    cosines = rows @ rows.T
    pairs = cosines[~np.eye(len(rows), dtype=bool)]
    assert users.loc[5, "semantic_similarity"] == pytest.approx(pairs.mean(), abs=1e-5)
    assert users.loc[5, "similarity_posts"] == 10
    # One measured comment has no pairs; empty comments have no vector.
    assert users.loc[4, "similarity_posts"] == 1
    assert pd.isna(users.loc[4, "semantic_similarity"])
    assert np.isnan(matrix[(enriched["text"] == "").to_numpy()]).all()


def test_rare_words_leave_similarity_unavailable(data_dir: Path) -> None:
    _, enriched = build_features(*read_corpus(data_dir))
    assert train_embeddings(enriched["text"], replace(SMALL, embedding_min_count=1000)) is None
    users = user_similarity(enriched, None)
    assert users["semantic_similarity"].isna().all()
    assert users["similarity_posts"].eq(0).all()
    assert lexicon_candidates(None, SMALL).empty


def test_lexicon_candidates_exclude_existing_terms(data_dir: Path) -> None:
    _, enriched = build_features(*read_corpus(data_dir))
    vectors = train_embeddings(enriched["text"], SMALL)
    candidates = lexicon_candidates(vectors, SMALL)
    assert set(candidates["side"]) == {"left-associated", "right-associated"}
    assert (
        not candidates["candidate"]
        .map(lambda word: any(p.fullmatch(word) for p in PATTERNS.values()))
        .any()
    )
    assert (
        candidates["seeds"]
        .str.split(";")
        .explode()
        .isin(
            ["geflüchteten", "geflüchtete", "schutzsuchende", "wirtschaftsflüchtlinge", "asylflut"]
        )
        .all()
    )
    assert (candidates["corpus_count"] > 0).all()
    own = candidates["left_similarity"].where(
        candidates["side"] == "left-associated", candidates["right_similarity"]
    )
    assert (
        (own - candidates["margin"])
        .round(6)
        .isin(candidates[["left_similarity", "right_similarity"]].stack().round(6).tolist())
        .all()
    )
    # Seeds rarer than the minimum count propose nothing.
    assert lexicon_candidates(vectors, replace(SMALL, lexicon_min_count=11)).empty
