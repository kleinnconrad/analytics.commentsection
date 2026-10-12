"""Installed command-line interface for exporting and analysing a corpus."""

import argparse
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from comment_analytics.benchmark import benchmark
from comment_analytics.config import FLAGS, Settings
from comment_analytics.data import export_database, file_hash, provenance, read_corpus, write_json
from comment_analytics.embeddings import lexicon_candidates, train_embeddings, user_similarity
from comment_analytics.evaluation import staff_overlap
from comment_analytics.features import build_features, detect
from comment_analytics.political import LEXICON, LEXICON_VERSION, aggregate_vocabulary
from comment_analytics.publish import publish
from comment_analytics.reporting import write_reports

LOG = logging.getLogger(__name__)


def run_pipeline(data_dir: Path, output: Path, settings: Settings) -> None:
    """Run analysis into a new directory without mixing or overwriting previous runs.

    Args:
        data_dir: Exported corpus CSV directory.
        output: New or empty destination directory.
        settings: Explicit behavioral and vocabulary thresholds.

    """
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new run directory")
    inputs = [data_dir / name for name in ("Posts.csv", "Articles.csv", "Newspaper_Staff.csv")]
    metadata = provenance(
        inputs,
        {
            "settings": asdict(settings),
            "lexicon_version": LEXICON_VERSION,
            "lexicon": LEXICON,
            "models": [],
            "status": "exploratory, not validated for intent",
        },
    )
    posts, articles = read_corpus(data_dir)
    features, enriched = build_features(posts, articles)
    LOG.info(
        "Training Word2Vec on %s comments with %s worker(s)",
        len(enriched),
        settings.embedding_workers,
    )
    vectors = train_embeddings(enriched["text"], settings)
    similarity = user_similarity(enriched, vectors)
    features = features.merge(similarity, on="ID_User", validate="one_to_one")
    users, cutoffs = detect(features, settings)
    political = aggregate_vocabulary(enriched, settings).drop(columns="total_posts")
    users = users.merge(political, on="ID_User", validate="one_to_one")
    users["writer_group"] = users["flag_any"].map({0: "Unflagged", 1: "Heuristic-flagged"})
    predictions = enriched.merge(
        users[["ID_User", *FLAGS, "flag_any", "writer_group", "vocabulary_group"]],
        on="ID_User",
        validate="many_to_one",
    )
    overlap = staff_overlap(users, pd.read_csv(data_dir / "Newspaper_Staff.csv"))
    output.mkdir(parents=True, exist_ok=True)
    users.to_csv(output / "users.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    write_json(output / "staff_overlap.json", overlap)
    write_reports(users, predictions, output, settings)
    lexicon_candidates(vectors, settings).to_csv(output / "lexicon_candidates.csv", index=False)
    if vectors is not None:
        vectors.save_word2vec_format(str(output / "word2vec.bin"), binary=True)
    metadata["cutoffs"] = cutoffs
    metadata["embeddings_deterministic"] = settings.embedding_workers == 1
    metadata["counts"] = {"posts": len(posts), "users": len(users)}
    metadata["outputs"] = {path.name: file_hash(path) for path in sorted(output.iterdir())}
    # Manifest is written last; its presence identifies a completed run.
    write_json(output / "manifest.json", metadata)
    LOG.info(
        "Analysed %s posts and %s users; report: %s", len(posts), len(users), output / "report.md"
    )


def main() -> None:
    """Parse explicit paths and run the requested operation with actionable errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export an existing database read-only")
    export.add_argument("--database", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run", help="Generate exploratory features, flags, and reports")
    run.add_argument("--data-dir", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--min-political-posts", type=int, default=Settings.min_political_posts)
    run.add_argument("--political-majority", type=float, default=Settings.political_majority)
    run.add_argument(
        "--embedding-workers",
        type=int,
        default=Settings.embedding_workers,
        help="Word2Vec threads; more than 1 is faster but not reproducible",
    )
    published = commands.add_parser(
        "publish", help="Overwrite the political comparison charts and pages from a completed run"
    )
    published.add_argument("--run", required=True, type=Path)
    published.add_argument("--output", required=True, type=Path)
    comparison = commands.add_parser("benchmark", help="Compare pinned models on reviewed examples")
    comparison.add_argument("--samples", type=Path, required=True)
    comparison.add_argument("--model", action="append", required=True)
    comparison.add_argument("--task", choices=["sentiment", "similarity"], required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "export":
            export_database(args.database, args.output)
        elif args.command == "publish":
            publish(args.run, args.output)
        elif args.command == "benchmark":
            benchmark(args.samples, args.model, args.task, args.output)
        else:
            run_pipeline(
                args.data_dir,
                args.output,
                Settings(
                    min_political_posts=args.min_political_posts,
                    political_majority=args.political_majority,
                    embedding_workers=args.embedding_workers,
                ),
            )
    except (OSError, ValueError, sqlite3.Error, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
