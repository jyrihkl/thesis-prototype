# Data

This folder contains project brief files and processed participant sets used by the prototype.

## Project sets

```text
projects_set_a.json
projects_set_b.json
projects_set_c.json
projects_set_d.json
projects_set_e.json
```

Run with:

```bash
team-builder --project-set a
team-builder --project-set e
```

## Participant sets

Generated participant sets are stored in:

```text
data/processed/participants/
```

Common files:

```text
candidates_080.csv
candidates_120.csv
candidates_240.csv
candidates_480.csv
candidates_1200.csv
candidates_2400.csv
candidate_sets_manifest.json
```

Run with:

```bash
team-builder --participant-set 240 --project-set c
```

The participant sets are derived from the `lang-uk/recruitment-dataset-candidate-profiles-english` dataset and are used only as prototype input data.
