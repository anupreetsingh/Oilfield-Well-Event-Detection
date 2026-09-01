# Oilfield Well-Event Detection & NPT Copilot

Predictive and generative AI on public oilfield data:

- well-event detection on the [Petrobras 3W](https://github.com/petrobras/3W) dataset: a PySpark
  pipeline that cleans and windows the runs into Delta tables, then classifiers for 8 fault types;
- non-productive time (NPT) on the Equinor Volve daily drilling reports: cause tagging and a
  copilot that drafts daily NPT summaries.

Work in progress.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[spark,dev]"
scripts/download_3w.sh          # ~1.7 GB of Parquet into data/raw/3w
```
