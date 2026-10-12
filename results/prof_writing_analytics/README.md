# Political vocabulary comparison

## Contents
- [Source](#source)
- [Method](#method)
- [Results](#results)
- [Limitations](#limitations)
- [Files](#files)

## Source

Run `main-lexicon-2026-10-12`, manifest SHA-256 `21374e6aef40`, Git revision `eef4a8db217f` (uncommitted changes), lexicon `main-crisis-v1`.
The run contains 1,011,773 posts and 31,413 users.
Regenerate these files with `comment-analytics publish --run <run directory> --output results/prof_writing_analytics`.

## Method

- **Comment label:** the comment body is matched against the lexicon `main-crisis-v1`
  (the "crisis" lexicon of the former main branch). The side with more distinct matched terms labels the comment.
- **User label:** at least 1 directional comment and at least 68% agreement. Users below the agreement threshold are mixed; users without directional comments are not labelled.
- **Groups:** "Heuristic-flagged" users meet at least one behavioral flag (temporal, topic, similarity, impact);
  all other users are "Unflagged". See the [root README](../../README.md#methodology).

## Results

2,422 of 31,413 users are left- or right-associated; 2,292 users are flagged.

| Flag group | Users | Left-associated | Right-associated | Left share | Right share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic-flagged | 2,292 | 209 | 105 | 66.6% | 33.4% |
| Unflagged | 29,121 | 1,433 | 675 | 68.0% | 32.0% |

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
