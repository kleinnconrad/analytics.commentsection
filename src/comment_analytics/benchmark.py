"""Opt-in model comparison on independently labelled German examples."""

import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from comment_analytics.data import file_hash, provenance, require_columns, write_json

SENTIMENTS = ["negative", "neutral", "positive"]


def sentiment_scores(expected: list[str], predicted: list[str]) -> dict[str, object]:
    """Compute a fixed-order confusion matrix and macro-F1 over all three sentiments."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have equal nonzero length")
    if not set(expected + predicted) <= set(SENTIMENTS):
        raise ValueError("Sentiment labels must be negative, neutral, or positive")
    matrix = (
        pd.crosstab(pd.Series(expected), pd.Series(predicted))
        .reindex(index=SENTIMENTS, columns=SENTIMENTS, fill_value=0)
        .to_numpy()
    )
    true_positives = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * true_positives, denominator, out=np.zeros(3), where=denominator > 0)
    return {
        "labels": SENTIMENTS,
        "confusion_matrix": matrix.tolist(),
        "support": matrix.sum(axis=1).tolist(),
        "macro_f1": float(f1.mean()),
    }


def benchmark(samples: Path, models: list[str], task: str, output: Path) -> None:
    """Compare immutable model revisions without using scores to classify users.

    Args:
        samples: Independently reviewed CSV. Sentiment requires text,label;
            similarity requires text_a,text_b,similarity with scores in [0, 1].
        models: Hugging Face model IDs followed by @ and a full revision SHA.
        task: sentiment or similarity.
        output: New or empty destination for per-example predictions and metrics.

    """
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output must be a new or empty directory")
    revisions = []
    for spec in models:
        model, separator, revision = spec.rpartition("@")
        if not separator or not model or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Each model must be MODEL_ID@FULL_40_CHARACTER_COMMIT_SHA")
        revisions.append((model, revision))
    data = pd.read_csv(samples)
    columns = {"text", "label"} if task == "sentiment" else {"text_a", "text_b", "similarity"}
    require_columns(data, columns, "Benchmark samples")
    if len(data) < 3 or data[list(columns)].isna().any().any():
        raise ValueError("Benchmark requires at least three complete reviewed examples")
    text_columns = ["text"] if task == "sentiment" else ["text_a", "text_b"]
    if any((data[column].astype(str).str.strip() == "").any() for column in text_columns):
        raise ValueError("Benchmark text must not be empty")
    if task == "sentiment":
        if set(data["label"]) != set(SENTIMENTS):
            raise ValueError("Sentiment evaluation requires all three sentiment classes")
    else:
        if not data["similarity"].between(0, 1).all() or data["similarity"].nunique() < 2:
            raise ValueError("Similarity labels must vary and lie in [0, 1]")
    metadata = provenance([samples], {"task": task, "models": models})
    metadata["packages"].update(
        {name: version(name) for name in ("sentence-transformers", "transformers", "torch")}
    )
    output.mkdir(parents=True, exist_ok=True)
    scores = []
    for index, (model_id, revision) in enumerate(revisions):
        result = data.copy()
        if task == "sentiment":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, revision=revision, trust_remote_code=False, use_safetensors=True
            )
            classifier = pipeline(
                "text-classification", model=model, tokenizer=tokenizer, device=-1
            )
            outputs = classifier(data["text"].tolist(), batch_size=16, truncation=True)
            labels = [item["label"].lower() for item in outputs]
            metrics = sentiment_scores(data["label"].tolist(), labels)
            result["prediction"] = labels
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_id,
                revision=revision,
                trust_remote_code=False,
                device="cpu",
                model_kwargs={"use_safetensors": True},
            )
            first = model.encode(data["text_a"].tolist(), normalize_embeddings=True)
            second = model.encode(data["text_b"].tolist(), normalize_embeddings=True)
            similarities = np.sum(first * second, axis=1)
            correlation = (
                float(spearmanr(data["similarity"], similarities).statistic)
                if np.unique(similarities).size > 1
                else None
            )
            metrics = {"spearman_correlation": correlation}
            result["prediction"] = similarities
        result.to_csv(output / f"model_{index}_predictions.csv", index=False)
        scores.append({"model": model_id, "revision": revision, "examples": len(data), **metrics})
    metadata["scores"] = scores
    metadata["limitations"] = (
        "Model comparison only; no automatic adoption or astroturfing validation."
    )
    metadata["outputs"] = {path.name: file_hash(path) for path in output.iterdir()}
    write_json(output / "manifest.json", metadata)
