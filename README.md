# Team Builder Prototype

Prototype decision-support pipeline for forming short-lived project teams from profile-based participant data and manually defined project briefs.

The system is intended to support human review. It does not predict full team effectiveness or replace final human judgment.

## Setup

```bash
python -m pip install -e .
```

## Prepare participant sets

The participant sets are included in the repository, but can be remade using the following.

```bash
python scripts/prepare_candidate_dataset.py \
  --sample-sizes 80 120 240 480 1200 2400 \
  --output-dir data/processed/participants \
  --write-legacy-default
```

## Run one recommendation

```bash
team-builder
```

Apart from manually set id, the above is equivalent to below.

```bash
team-builder --participant-set 080 --project-set a --run-id first-test
```

Other useful examples:

```bash
team-builder --participant-set 240 --project-set e
team-builder --participant-set 1200 --project-set c --no-save-run
team-builder --participants data/processed/participants/candidates_480.csv --project-set d
```

To view options:

```bash
team-builder --help
```

## Run batch evaluation

```bash
team-builder-batch --participant-set 240
```

Compare weight profiles:

```bash
team-builder-batch \
  --participant-set 240 \
  --weight-profiles default skill_heavy role_heavy balance_heavy
```

## Plot batch results

```bash
python scripts/plot_batch_results.py \
  --batch-dir runs/batches/<batch_id>
```

## Create a review template

```bash
python scripts/create_review_template.py \
  --allocation runs/first-test/allocation.json \
  --output reviews/first-test-review.csv \
  --run-id first-test \
  --project-set a \
  --participant-set 080 \
  --weight-profile default
```
