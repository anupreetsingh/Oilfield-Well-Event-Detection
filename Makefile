PY ?= .venv/bin/python
SYN = data/raw/volve_synthetic
SYN_SET = raw_dir=$(SYN) out_dir=data/npt_synthetic model_dir=models/npt-synthetic reports_dir=data/npt_synthetic/reports experiment=oilfield-npt-synthetic

.PHONY: setup lint test data-3w pipeline train-events train-npt tag-npt copilot summary bi demo-npt mlflow-ui

setup:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[all]"

lint:
	$(PY) -m ruff check src tests notebooks

test:
	$(PY) -m pytest -q

# --- 3W well events -----------------------------------------------------------------------
data-3w:
	scripts/download_3w.sh data/raw/3w

pipeline:                         ## raw 3W runs -> silver + gold Delta tables
	$(PY) -m oilfield.threew.pipeline --stage all

train-events:                     ## held-out-well CV for rf, hgb, cnn, ensemble -> reports/event_metrics.md
	$(PY) -m oilfield.events.train

# --- Volve NPT ------------------------------------------------------------------------------
train-npt:                        ## needs the Volve DDR XML under data/raw/volve/ddr
	$(PY) -m oilfield.npt.train

tag-npt:
	$(PY) -m oilfield.npt.tag

copilot:
	$(PY) -m oilfield.copilot.cli build

summary:                          ## make summary WELLBORE=F-12 DATE=2008-02-14 [OFFLINE=--offline]
	$(PY) -m oilfield.copilot.cli summary --wellbore $(WELLBORE) --date $(DATE) $(OFFLINE)

bi:
	$(PY) -m oilfield.bi.export

demo-npt:                         ## the whole NPT chain on synthetic reports, no downloads or API key
	$(PY) -m oilfield.npt.synthetic --out $(SYN)
	$(PY) -m oilfield.npt.train --models tfidf --set $(SYN_SET)
	$(PY) -m oilfield.npt.tag --set out_dir=data/npt_synthetic model_dir=models/npt-synthetic
	$(PY) -m oilfield.copilot.cli build --set npt_dir=data/npt_synthetic db_url=sqlite:///data/npt_synthetic/npt.db
	$(PY) -m oilfield.copilot.cli daily --date 2014-06-29 --offline \
		--set db_url=sqlite:///data/npt_synthetic/npt.db summaries_dir=data/npt_synthetic/summaries
	$(PY) -m oilfield.bi.export --npt-dir data/npt_synthetic --out data/npt_synthetic/powerbi

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db
