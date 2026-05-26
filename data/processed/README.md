# Processed data

This folder contains derived participant samples created for prototype evaluation.

Generate or refresh the samples with:

```bash
python scripts/prepare_candidate_dataset.py \
  --sample-sizes 80 120 240 480 1200 2400 \
  --output-dir data/processed/participants \
  --write-legacy-default
```

The legacy files below are kept for compatibility with earlier prototype runs:

```text
candidates_filtered.csv
candidates_filtered.jsonl
candidates_filter_report.json
```

Preferred participant files are in:

```text
participants/candidates_080.csv
participants/candidates_120.csv
participants/candidates_240.csv
participants/candidates_480.csv
participants/candidates_1200.csv
participants/candidates_2400.csv
```

## Source and attribution

The processed files are derived from `lang-uk/recruitment-dataset-candidate-profiles-english`, the English candidate-profile subset of the Djinni Recruitment Dataset hosted on Hugging Face.

Original dataset paper:

Drushchak, N., & Romanyshyn, M. (2024). *Introducing the Djinni Recruitment Dataset: A corpus of anonymized CVs and job postings*. Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024. https://aclanthology.org/2024.unlp-1.2/

Dataset page: https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english

## Notes

These files are a derived subset prepared for prototype development and evaluation. They should be regenerated with `scripts/prepare_candidate_dataset.py` if the filtering criteria or skill vocabulary changes.
