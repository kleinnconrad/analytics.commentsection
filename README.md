# Comment Section Analytics

Exploratory analysis of the German One Million Posts Corpus. Behavioral flags and vocabulary associations are descriptive indicators, not evidence of employment, political identity, coordination, or astroturfing.

## Contents
- [Dataset](#dataset)
- [Setup and execution](#setup-and-execution)
- [Methodology](#methodology)
  - [Word embeddings](#word-embeddings)
- [Outputs](#outputs)
- [Model evaluation](#model-evaluation)
- [Pre-merge checks](#pre-merge-checks)
- [Migration](#migration)
- [Credits and license](#credits-and-license)

## Dataset

The [publisher](https://ofai.github.io/million-post-corpus/) describes a collection window of **June 1, 2015 to May 31, 2016**, comprising 1,011,773 posts and 12,087 articles from *Der Standard*. Article dates or timestamp extrema do not establish 13 years of continuous comment coverage. Each run reports its actual post timestamp extrema separately.

Download and decompress the SQLite corpus from the publisher's site, subject to its license, and place it at `db/corpus.sqlite3`. The database is not included in Git. See [the database schema](db/database_schema.md). `Newspaper_Staff` identifies a subset of users; it is not a parent table containing every post author.

## Setup and execution

Use Python 3.11, 3.12, or 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/). Dependencies are declared in `pyproject.toml`; `uv.lock` records the resolved versions and hashes. CI uses uv 0.12.9. No Conda environment or requirements file is needed.

```bash
uv sync --locked --python 3.12
uv run --locked comment-analytics export --database db/corpus.sqlite3 --output data/corpus
uv run --locked comment-analytics run --data-dir data/corpus --output results/runs/baseline
```

The installed `comment-analytics` command works outside the repository too. Paths are explicit, so use absolute paths when invoking it from another directory. Export and analysis destinations must be new or empty; existing runs are never overwritten. The SQLite database is opened read-only and exported in chunks. Analysis loads the post and article tables into memory. On the full corpus (1,011,773 posts, Windows 11, Python 3.11), export took 13 s; analysis took 16 minutes with a peak working set of 3.2 GB, of which Word2Vec training on one thread took 14 minutes. `--embedding-workers 4` reduced the run to 7 minutes but makes the vectors and similarity flags differ between runs.

The analysis requires `Posts.csv`, `Articles.csv`, and `Newspaper_Staff.csv`. Invalid keys, missing article references, malformed dates, and invalid votes cause errors rather than silently dropped rows. Headline and body are combined. Empty text remains unmeasured, not a neutral sentiment observation. Business hours use the supplied timestamps; their source timezone must be established before interpreting this feature.

## Methodology

Thresholds live in [Settings](src/comment_analytics/config.py) and are included in every generated report. All flags require the configured minimum post count:

- **Temporal:** sufficient posts per active day, a high weekday 09:00-17:59 share, and low average article-to-comment delay. Negative delays are excluded, and affected users cannot satisfy this flag.
- **Topic concentration:** low Shannon entropy in nats across the second component of `Newsroom/...` paths, with sufficient posts and no missing topic assignments.
- **Semantic similarity:** mean pairwise cosine similarity of a user's comment vectors, strictly above the configured quantile among users with sufficient posts and at least that many measured comments. Cutoff ties are excluded. See [Word embeddings](#word-embeddings).
- **Impact:** average `positive_votes * negative_votes / (positive_votes + negative_votes + 1)` strictly above the configured quantile among eligible users. Cutoff ties are excluded, so a constant distribution flags nobody.

`flag_any` is the OR of these indicators. It is neither a probability nor a validated classifier. `upvotes_per_article_delay` is retained descriptively but does not influence flags; there are no vote timestamps from which to infer voting speed or coordination. English VADER, capitalization-based formality, and low-variance sentiment no longer determine flags. `duplicate_share` (`1 - unique_normalized_texts / nonempty_text_posts`) is retained descriptively.

### Word embeddings

The similarity flag uses [gensim](https://radimrehurek.com/gensim/) Word2Vec (skip-gram), trained on the analysed comments themselves. No pretrained model is downloaded.

- **Tokens:** lowercase letter sequences. Digits, punctuation, and underscores separate tokens, so comments that differ only in numbers are identical.
- **Training:** 100 dimensions, window 5, minimum count 5, 5 epochs, seed 1. Words below the minimum count are ignored.
- **Comment vector:** the mean of the unit word vectors of its known words. The corpus mean comment vector is subtracted before normalization. Without this step, averaged word vectors of unrelated German comments have high cosine similarity, because they share a common component. Comments without known words are unmeasured.
- **User similarity:** the mean cosine similarity over all distinct pairs of a user's measured comments, computed exactly as `(|sum|^2 - n) / (n (n - 1))` for unit vectors, without sampling. Values are rounded to six decimals so that float noise does not separate tied users.
- **Reproducibility:** with one worker thread (default), training is deterministic for a fixed seed: separate runs produce identical vectors. `--embedding-workers N` trains faster, but vectors and flags then differ between runs; the manifest records `embeddings_deterministic`.

High similarity can result from copied text, templates, or a narrow topic. It is not evidence of coordination.

The trained vectors also propose lexicon extensions in `lexicon_candidates.csv`:

- **Seeds:** single-word vocabulary entries that the lexicon matches and that occur at least 20 times. Multi-word terms and rarer words are not used, because vectors of rare words are unreliable.
- **Candidates:** the 20 nearest neighbours of each seed that occur at least 20 times and are not already matched by the lexicon.
- **Columns:** `left_similarity` and `right_similarity` are the highest cosine similarity to any seed of each side. `margin` is the similarity to the proposing side minus the similarity to the other side. Topic words close to both sides, such as `migranten`, have a margin near or below zero.

Neighbours share contexts, which often means the same topic rather than the same stance. Candidates are never added automatically; review them, edit `LEXICON` in [political.py](src/comment_analytics/political.py), and increase `LEXICON_VERSION`.

Political analysis uses the "crisis" lexicon and the matching rules of the former `main` branch (`LEXICON_VERSION = "main-crisis-v1"` in [political.py](src/comment_analytics/political.py)):

- **Text:** the comment body, lowercased. Headlines are not classified.
- **Matching:** each term matches as a whole word in the listed form only. Inflections such as `geflüchteten` or `wirtschaftsflüchtling` do not match.
- **Comment label:** the side with more distinct matched terms. Ties and comments without matches are unknown.
- **No interpretation:** negation, quotation, sarcasm, and context are not handled. "Keine Asylflut" counts as right-associated.

On the full corpus, generic terms (`menschenrechte`, `menschlichkeit`, `stammtisch`, `hassposter`, `obergrenze`, `fachkräfte`, `kontrollverlust`) produce about 4,000 of the 5,262 directional comments. Without them, the left/right ratio of directional comments reverses. Interpret left/right shares with this in mind.

User labels describe **left-associated or right-associated vocabulary**, not the user's ideology. As on the former `main` branch, one directional comment is enough by default, and at least 68% of a user's directional comments must agree (inclusive). Users below the agreement threshold are `mixed`; users without directional comments are `insufficient-evidence`. Unknown text is not evidence of political neutrality. The thresholds are exploratory, not a sigma rule or significance test.

With the default of one comment, a single word decides most labels: of 2,422 directional users, 27 have at least 10 directional comments. Override the thresholds with `--min-political-posts` and `--political-majority`; `threshold_sensitivity.csv` lists directional user counts for minimums of 1, 3, 5, 10, and 20 comments.

Staff evaluation reports counts and overlap rates only. Nonstaff users are not verified negative examples of professional status. Undefined rates are JSON `null`, not zero.

## Outputs

Each completed run contains:

| File | Purpose |
| --- | --- |
| `users.csv` | Features, similarity, flags, vocabulary counts, and coverage per user |
| `predictions.csv` | Every input post, including its ID, combined text, path, and user flags |
| `staff_overlap.json` | User-level staff overlap counts and rates |
| `user_shares.csv` | Label counts and shares using all users in each flag group; `directional_share` uses only left- and right-associated users, as charted on the former `main` branch |
| `path_comment_shares.csv` | Counts, unique-user counts, and activity-weighted comment shares by observed year, path, and group |
| `threshold_sensitivity.csv` | Directional user counts across minimum-evidence and majority thresholds |
| `vocabulary_shares.png`, `report.md` | Descriptive visualization and interpretation limits |
| `word2vec.bin` | gensim Word2Vec vectors trained on the input (binary word2vec format); absent when no word reaches the minimum count |
| `lexicon_candidates.csv` | Nearest-neighbour words of lexicon terms, for manual review |
| `manifest.json` | Input/output SHA-256 hashes, source hashes, Git identity when available, installed versions, thresholds, and lexicon |

The manifest is written last; an output directory without it is incomplete. A dirty checkout is recorded explicitly, with source hashes to identify its code. No pretrained models participate in the default analysis; the Word2Vec vectors are trained on the input and saved with the run.

User charts include insufficient-evidence users in their denominators. Path shares count all comments by users with a given vocabulary label, including nonpolitical comments, and are weighted by activity. They are not targeting estimates. Neither population confidence intervals nor political accuracy estimates are justified without an appropriate sampling and labelling study. Existing files under [results/prof_writing_analytics](results/prof_writing_analytics/README.md) are historical, unvalidated outputs, not results of this pipeline.

## Model evaluation

Model selection remains pending independent German-language evaluation. Install optional tooling with `uv sync --locked --extra nlp`. The benchmark runs locally on CPU and downloads model weights; it never enables a model in the detection pipeline.

Prepare a separate reviewed CSV, retain post IDs and annotation provenance as extra columns, and reserve a held-out sample before tuning. For sentiment, require `text,label`, with labels `negative`, `neutral`, and `positive`. Corpus sentiment annotations can support sentiment evaluation, but not astroturfing or political-stance evaluation. For similarity, require `text_a,text_b,similarity` with reviewed scores in [0, 1]. Do not treat lexicon-generated labels as independent ground truth.

```bash
uv run --locked --extra nlp comment-analytics benchmark \
  --task sentiment --samples data/reviewed_sentiment.csv \
  --model "MODEL_ID@FULL_40_CHARACTER_COMMIT_SHA" \
  --model "SECOND_MODEL_ID@FULL_40_CHARACTER_COMMIT_SHA" \
  --output results/runs/sentiment-comparison
```

Replace the placeholders with German-capable model IDs and immutable revisions from their model repositories. Sentiment models must expose the three label names above; generic `LABEL_0` mappings are rejected rather than guessed. Model weights must use safetensors; remote model code is disabled. Candidates to evaluate include [German sentiment BERT](https://huggingface.co/oliverguhr/german-sentiment-bert) and [multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) for their respective tasks; neither is an endorsed replacement without evaluation.

Repeat with `--task similarity` for embedding comparison. Outputs include per-example predictions, confusion matrices and macro-F1 for sentiment, or Spearman correlation for similarity, plus model revisions and input hashes. Three examples are merely the input minimum, not an adequate validation sample. Review errors across negation, quotation, inflection, and sarcasm; report sample sizes, class balance, uncertainty, and held-out results before adoption. No real-corpus model comparison has yet been completed.

## Pre-merge checks

Run the same checks locally:

```bash
uv lock --check
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked actionlint
uv run --locked pytest
uv build
```

The [Pre-merge workflow](.github/workflows/pre-merge.yml) runs on pull requests, main-branch pushes, and merge queues. It validates the lockfile, code style, workflow expressions, package builds, and tests on Python 3.11-3.13. Tests install a non-editable package and exercise a disposable SQLite-to-report run from an unrelated directory, including zero-evidence groups, denominators, input validation, output hashes, and refusal to overwrite results. They do not need the corpus, secrets, model downloads, or a GPU. JUnit results are retained for seven days.

Actions are pinned to commit IDs and use read-only permissions. Dependabot updates uv dependencies and GitHub Actions; it no longer automatically approves its own PRs. Configure **Pre-merge gate** as a required status check in the main-branch ruleset after the workflow is pushed and has run. The workflow alone does not enforce branch protection, and this change does not modify remote rulesets. Full-corpus accuracy, performance, and model evaluation are separate validation steps, not claims made by green CI.

## Migration

The numbered scripts, `db/export_data.py`, duplicate report implementations, and old lexicon switch have been replaced by the installed CLI. The "crisis" lexicon is kept unchanged; the unused larger lexicon was removed. Old prediction CSVs are not inputs to the new pipeline; regenerate from the original corpus export. The `is_professional_*` columns have been replaced with `flag_*`, and the old content rule (English sentence-transformer similarity on a sample of 50 posts, VADER, and formality) has been replaced by gensim Word2Vec similarity over all comments. Results will change and should not be compared as if they used the same method.

## Credits and license

The corpus was collected and annotated by Dietmar Schabus, Marcin Skowron, Martin Trapp, and collaborators at OFAI and *Der Standard*. Cite *One Million Posts: A Data Set of German Online Discussions*, SIGIR 2017, [DOI:10.1145/3077136.3080711](https://doi.org/10.1145/3077136.3080711).

The dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); respect its noncommercial and attribution conditions. See [LICENSE](LICENSE) for repository licensing. Do not commit corpus exports or identifiable post-level results without checking the applicable terms.
