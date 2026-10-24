# Run the whole study end-to-end:  make all
PY=python
export PYTHONPATH=src

install:
	pip install -e ".[dl,dev]"
data:
	$(PY) scripts/01_generate_data.py
train:
	$(PY) scripts/02_train_tabular.py
	$(PY) scripts/03_train_sequence.py
evaluate:
	$(PY) scripts/04_evaluate.py
	$(PY) scripts/05_make_figures.py
test:
	pytest -q
serve:
	uvicorn ews.serving.app:app --reload --port 8000
all: data train evaluate
