# Processed data

This folder contains the cleaned candidate subset used by the team-formation prototype.

## Contents

- `candidates_filtered.csv`: tabular candidate pool after filtering, role-family inference, and skill extraction.
- `candidates_filtered.jsonl`: the same processed candidate pool in JSON Lines format.
- `candidates_filter_report.json`: summary of filtering settings and retained records.

## Source and attribution

The processed files are derived from `lang-uk/recruitment-dataset-candidate-profiles-english`, the English candidate-profile subset of the Djinni Recruitment Dataset hosted on Hugging Face.

Original dataset paper:

Drushchak, N., & Romanyshyn, M. (2024). *Introducing the Djinni Recruitment Dataset: A corpus of anonymized CVs and job postings*. Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024. https://aclanthology.org/2024.unlp-1.2/

Dataset page: https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english

## Notes

These files are a derived subset prepared for prototype development and evaluation. They should be regenerated with `scripts/prepare_candidate_dataset.py` if the filtering criteria or skill vocabulary changes.
