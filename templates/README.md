# Templates

Templates for review-based evaluation.

## Review template

```text
review_template.csv
review_instructions.md
```

The review form asks evaluators to rate each recommended team on:

```text
usefulness
fairness
transparency
team fit
acceptability
```

Use the helper script to create a pre-filled review form from a saved allocation:

```bash
python scripts/create_review_template.py \
  --allocation runs/first-test/allocation.json \
  --output reviews/first-test-review.csv
```
