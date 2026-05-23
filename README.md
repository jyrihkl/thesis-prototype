# thesis-prototype

## Project initialization

The package uses a `src/` layout. Install it once in editable mode from the repository root:

```bash
python -m pip install -e .
```

After that, either command works:

```bash
python main.py
team-builder
```

Use project set B:

```bash
team-builder --project-set b
```

Pass explicit files:

```bash
team-builder \
  --participants data/processed/candidates_filtered.csv \
  --projects data/projects_set_a.json
```