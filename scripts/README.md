# Scripts

Utility scripts used outside the main pipeline.

## Prepare participant sets

```bash
python scripts/prepare_candidate_dataset.py \
  --sample-sizes 80 120 240 480 1200 2400 \
  --output-dir data/processed/participants \
  --write-legacy-default
```

## Plot batch results

```bash
python scripts/plot_batch_results.py \
  --batch-dir runs/batches/<batch_id>
```

## Create a review form

```bash
python scripts/create_review_template.py \
  --allocation runs/first-test/allocation.json \
  --output reviews/first-test-review.csv \
  --run-id first-test \
  --project-set a \
  --participant-set 080 \
  --weight-profile default
```
