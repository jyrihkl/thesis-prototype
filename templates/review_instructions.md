# Review template

This template supports the human-centered evaluation part of the prototype. It is intentionally kept outside the main recommendation pipeline so that review remains a separate evaluation activity rather than part of the algorithm.

## Rating scale

Use a 1 to 5 scale:

```text
1 = very poor
2 = poor
3 = acceptable / mixed
4 = good
5 = very good
```

## Rating dimensions

- `perceived_usefulness_1_5`: How useful is the suggested team for the project?
- `perceived_fairness_1_5`: Does the team seem fair and reasonable compared with the other teams in the same run?
- `perceived_transparency_1_5`: Is the recommendation understandable from the provided explanation fields?
- `perceived_team_fit_1_5`: Does the team appear to fit the project requirements?
- `would_accept_team_yes_no`: Would the reviewer accept this team as suggested?

## Free-text fields

- `suggested_changes`: Any concrete changes the reviewer would make.
- `comments`: Any broader concerns, trade-offs, or interpretation notes.

## Suggested use

For a small thesis evaluation, one reviewer can rate each recommended team from a selected set of runs. If multiple reviewers are used, `reviewer_id_optional` can be filled with anonymous labels such as `R1`, `R2`, and `R3`.

The template does not validate actual later team effectiveness. It evaluates whether the recommendation appears useful, fair, transparent, and plausible as decision support.
