"""Train corpus Word2Vec embeddings with gensim and measure per-user comment similarity."""

import logging
import re
from collections.abc import Iterator

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from scipy import sparse

from comment_analytics.config import Settings
from comment_analytics.political import LEXICON

LOG = logging.getLogger(__name__)
TOKEN = re.compile(r"[^\W\d_]+")
CHUNK_SIZE = 50_000


def tokenize(text: str) -> list[str]:
    """Return lowercase letter-only tokens; digits, punctuation, and underscores split words."""
    return TOKEN.findall(text.lower())


class TokenStream:
    """Re-iterable token stream, so gensim can make several passes without holding all tokens."""

    def __init__(self, texts: pd.Series) -> None:
        """Store the comment texts to tokenize on each pass."""
        self.texts = texts

    def __iter__(self) -> Iterator[list[str]]:
        """Yield one token list per comment."""
        return (tokenize(text) for text in self.texts)


def train_embeddings(texts: pd.Series, settings: Settings) -> KeyedVectors | None:
    """Train skip-gram Word2Vec on the analysed comments.

    With one worker thread, training is deterministic for a fixed seed. More
    threads are faster, but vectors then differ between runs.

    Args:
        texts: Combined headline and body text of every post.
        settings: Embedding dimensions, window, minimum count, epochs, seed, and workers.

    Returns:
        Trained word vectors, or None when no word reaches the minimum count.

    """
    model = Word2Vec(
        sg=1,
        vector_size=settings.embedding_dimensions,
        window=settings.embedding_window,
        min_count=settings.embedding_min_count,
        epochs=settings.embedding_epochs,
        seed=settings.embedding_seed,
        workers=settings.embedding_workers,
    )
    stream = TokenStream(texts)
    model.build_vocab(stream)
    if len(model.wv) == 0:
        LOG.warning("No word reaches the embedding minimum count; similarity is unavailable")
        return None
    model.train(stream, total_examples=model.corpus_count, epochs=model.epochs)
    return model.wv


def comment_vectors(texts: pd.Series, vectors: KeyedVectors) -> np.ndarray:
    """Average unit word vectors per comment, remove the corpus mean, and normalize.

    Averaged word vectors share a common component, so uncentered cosine
    similarities between unrelated comments are high. Subtracting the mean
    comment vector removes it. Comments without known words remain NaN.

    Args:
        texts: Comment texts in post order.
        vectors: Trained word vectors.

    Returns:
        A float32 matrix with one unit-length row per comment, or NaN rows.

    """
    normed = vectors.get_normed_vectors()
    index = vectors.key_to_index
    result = np.full((len(texts), normed.shape[1]), np.nan, dtype=np.float32)
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts.iloc[start : start + CHUNK_SIZE]
        rows, columns = [], []
        for row, text in enumerate(chunk):
            known = [index[token] for token in tokenize(text) if token in index]
            rows.extend([row] * len(known))
            columns.extend(known)
        counts = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, columns)),
            shape=(len(chunk), len(normed)),
        )
        tokens = np.asarray(counts.sum(axis=1)).ravel()
        means = (counts @ normed) / np.where(tokens > 0, tokens, 1)[:, None]
        means[tokens == 0] = np.nan
        result[start : start + len(chunk)] = means
    measured = ~np.isnan(result).any(axis=1)
    if measured.any():
        centered = result[measured] - result[measured].mean(axis=0)
        # A comment exactly at the corpus mean has no direction and becomes NaN (0 / 0).
        with np.errstate(invalid="ignore"):
            result[measured] = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    return result


def user_similarity(posts: pd.DataFrame, vectors: KeyedVectors | None) -> pd.DataFrame:
    """Return each user's mean pairwise cosine similarity over all measured comments.

    For unit vectors, the mean over all distinct pairs equals
    (|sum|^2 - n) / (n * (n - 1)), so no sampling or quadratic comparison is needed.

    Args:
        posts: Enriched posts containing ID_User and text.
        vectors: Trained word vectors, or None when training was not possible.

    Returns:
        One row per user with similarity_posts and semantic_similarity (NaN below two posts).

    """
    users = pd.DataFrame({"ID_User": np.sort(posts["ID_User"].unique())})
    if vectors is None:
        return users.assign(similarity_posts=0, semantic_similarity=np.nan)
    matrix = comment_vectors(posts["text"], vectors)
    measured = ~np.isnan(matrix).any(axis=1)
    frame = pd.DataFrame(matrix[measured].astype(np.float64))
    frame["ID_User"] = posts["ID_User"].to_numpy()[measured]
    grouped = frame.groupby("ID_User")
    sums = grouped.sum()
    counts = grouped.size()
    squared = (sums.to_numpy() ** 2).sum(axis=1)
    similarity = (squared - counts) / (counts * (counts - 1)).where(counts > 1)
    # Rounding keeps float noise from separating users whose comments are equally similar.
    result = pd.DataFrame(
        {"similarity_posts": counts, "semantic_similarity": similarity.clip(-1, 1).round(6)}
    ).reset_index()
    users = users.merge(result, on="ID_User", how="left", validate="one_to_one")
    users["similarity_posts"] = users["similarity_posts"].fillna(0).astype(int)
    return users


def lexicon_candidates(vectors: KeyedVectors | None, settings: Settings) -> pd.DataFrame:
    """Propose nearest-neighbour words of lexicon terms for manual review.

    Neighbours share contexts, which often means the same topic rather than the
    same stance. Rare words have unreliable vectors, so seeds and candidates need
    a minimum corpus count. The margin is the similarity to the proposing side's
    seeds minus the similarity to the other side's seeds; words close to both
    sides have a margin near zero. Candidates are never added to the lexicon.

    Args:
        vectors: Trained word vectors, or None when training was not possible.
        settings: Neighbours per seed and the minimum corpus count.

    Returns:
        One row per side and candidate, sorted by margin within each side.

    """
    columns = [
        "side",
        "candidate",
        "margin",
        "left_similarity",
        "right_similarity",
        "seeds",
        "corpus_count",
    ]
    if vectors is None:
        return pd.DataFrame(columns=columns)
    known = {
        side: re.compile("|".join(re.escape(term) for term in terms))
        for side, terms in LEXICON.items()
    }

    def frequent(word: str) -> bool:
        return vectors.get_vecattr(word, "count") >= settings.lexicon_min_count

    seeds = {
        side: [word for word in vectors.index_to_key if pattern.fullmatch(word) and frequent(word)]
        for side, pattern in known.items()
    }
    for side, words in seeds.items():
        missing = [term for term in LEXICON[side] if term not in words]
        if missing:
            LOG.info("No frequent single-word seed for %s terms: %s", side, missing)
    if not all(seeds.values()):
        LOG.warning("Lexicon candidates need frequent seeds on both sides; none proposed")
        return pd.DataFrame(columns=columns)
    rows = []
    for side, words in seeds.items():
        for seed in words:
            for word, _ in vectors.most_similar(seed, topn=settings.lexicon_neighbors):
                if (
                    len(word) >= 3
                    and frequent(word)
                    and not any(pattern.fullmatch(word) for pattern in known.values())
                ):
                    rows.append({"side": side, "candidate": word, "seed": seed})
    if not rows:
        return pd.DataFrame(columns=columns)
    table = (
        pd.DataFrame(rows)
        .groupby(["side", "candidate"])["seed"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("seeds")
        .reset_index()
    )
    for side in ("left", "right"):
        table[f"{side}_similarity"] = [
            float(1 - vectors.distances(word, seeds[f"{side}-associated"]).min())
            for word in table["candidate"]
        ]
    own = table["left_similarity"].where(
        table["side"] == "left-associated", table["right_similarity"]
    )
    other = table["right_similarity"].where(
        table["side"] == "left-associated", table["left_similarity"]
    )
    table["margin"] = own - other
    table["corpus_count"] = [vectors.get_vecattr(word, "count") for word in table["candidate"]]
    return table.sort_values(["side", "margin"], ascending=[True, False])[columns]
