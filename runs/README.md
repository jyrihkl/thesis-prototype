# Runs

Saved single-run and batch outputs are written here.

## Single run

```bash
team-builder --participant-set 080 --project-set a --run-id first-test
```

Creates:

```text
runs/first-test/
  report.txt
  run_summary.json
  allocation.json
  baseline_comparison.json
```

## Batch run

```bash
team-builder-batch --participant-set 240
```

Creates:

```text
runs/batches/<batch_id>/
  batch_runs.csv
  batch_methods.csv
  batch_summary.json
  batch_report.txt
  runs/
```

Plot a batch with:

```bash
python scripts/plot_batch_results.py \
  --batch-dir runs/batches/<batch_id>
```
