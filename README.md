# Comment Section Analytics

Exploratory analysis of the German One Million Posts Corpus. Behavioral flags and vocabulary associations are descriptive indicators, not evidence of employment, political identity, coordination, or astroturfing.

## Contents
- [Dataset](#dataset)
- [Setup and execution](#setup-and-execution)
- [Methodology](#methodology)
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

The installed `comment-analytics` command works outside the repository too. Paths are explicit, so use absolute paths when invoking it from another directory. Export and analysis destinations must be new or empty; existing runs are never overwritten. The SQLite database is opened read-only and exported in chunks. Analysis currently loads the post and article tables into memory; full-corpus memory and runtime measurements remain pending.

The analysis requires `Posts.csv`, `Articles.csv`, and `Newspaper_Staff.csv`. Invalid keys, missing article references, malformed dates, and invalid votes cause errors rather than silently dropped rows. Headline and body are combined. Empty text remains unmeasured, not a neutral sentiment observation. Business hours use the supplied timestamps; their source timezone must be established before interpreting this feature.

## Methodology

Thresholds live in [Settings](src/comment_analytics/config.py) and are included in every generated report. All flags require the configured minimum post count:

- **Temporal:** sufficient posts per active day, a high weekday 09:00-17:59 share, and low average article-to-comment delay. Negative delays are excluded, and affected users cannot satisfy this flag.
- **Topic concentration:** low Shannon entropy in nats across the second component of `Newsroom/...` paths, with sufficient posts and no missing topic assignments.
- **Exact repetition:** `1 - unique_normalized_texts / nonempty_text_posts`. Normalization folds case and collapses whitespace. Empty text is excluded. Similarity of meaning is not measured.
- **Impact:** average `positive_votes * negative_votes / (positive_votes + negative_votes + 1)` strictly above the configured quantile among eligible users. Cutoff ties are excluded, so a constant distribution flags nobody.

`flag_any` is the OR of these indicators. It is neither a probability nor a validated classifier. `upvotes_per_article_delay` is retained descriptively but does not influence flags; there are no vote timestamps from which to infer voting speed or coordination. English VADER, capitalization-based formality, and low-variance sentiment no longer determine flags.

Political analysis uses one versioned, explicit-inflection lexicon. Ambiguous everyday terms were removed. Each lexicon concept counts at most once per comment. Detected quotation or negation causes abstention, as do ties and unmatched text. The matcher cannot reliably interpret sarcasm, indirect quotation, or context.

User labels describe **left-associated or right-associated vocabulary**, not the user's ideology. A minimum number of directional matches and an inclusive agreement threshold are required. Otherwise users remain `mixed` or `insufficient-evidence`; unknown text is not evidence of political neutrality. The thresholds are exploratory, not a sigma rule or significance test. Override political thresholds with `--min-political-posts` and `--political-majority`; inspect sensitivity before interpreting differences.

Staff evaluation reports counts and overlap rates only. Nonstaff users are not verified negative examples of professional status. Undefined rates are JSON `null`, not zero.

## Outputs

Each completed run contains:

| File | Purpose |
| --- | --- |
| `users.csv` | Features, flags, vocabulary counts, and coverage per user |
| `predictions.csv` | Every input post, including its ID, combined text, path, and user flags |
| `staff_overlap.json` | User-level staff overlap counts and rates |
| `user_shares.csv` | Label counts and shares using all users in each flag group |
| `path_comment_shares.csv` | Counts, unique-user counts, and activity-weighted comment shares by observed year, path, and group |
| `threshold_sensitivity.csv` | Directional user counts across minimum-evidence and majority thresholds |
| `vocabulary_shares.png`, `report.md` | Descriptive visualization and interpretation limits |
| `manifest.json` | Input/output SHA-256 hashes, source hashes, Git identity when available, installed versions, thresholds, and lexicon |

The manifest is written last; an output directory without it is incomplete. A dirty checkout is recorded explicitly, with source hashes to identify its code. No pretrained models participate in the default analysis.

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

The numbered scripts, `db/export_data.py`, duplicate report implementations, and old lexicon switch have been replaced by the installed CLI. Old prediction CSVs are not inputs to the new pipeline; regenerate from the original corpus export. The `is_professional_*` columns have been replaced with `flag_*`, and the old content rule has been replaced by exact repetition. Results will change and should not be compared as if they used the same method.

## Credits and license

The corpus was collected and annotated by Dietmar Schabus, Marcin Skowron, Martin Trapp, and collaborators at OFAI and *Der Standard*. Cite *One Million Posts: A Data Set of German Online Discussions*, SIGIR 2017, [DOI:10.1145/3077136.3080711](https://doi.org/10.1145/3077136.3080711).

The dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); respect its noncommercial and attribution conditions. See [LICENSE](LICENSE) for repository licensing. Do not commit corpus exports or identifiable post-level results without checking the applicable terms.
