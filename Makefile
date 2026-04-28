PYTHON = venv/bin/python
PIP    = venv/bin/pip

.PHONY: help ingest serve ui eval install

help:
	@echo ""
	@echo "  LegalDocAssistant — available commands"
	@echo "  ─────────────────────────────────────"
	@echo "  make install   Install all dependencies"
	@echo "  make ingest    Index documents in data/documents/"
	@echo "  make serve     Start FastAPI backend (port 8000)"
	@echo "  make ui        Start Streamlit frontend (port 8501)"
	@echo "  make eval      Run evaluation pipeline"
	@echo ""

install:
	python3.13 -m venv venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) -m backend.ingestion

serve:
	venv/bin/uvicorn backend.main:app --reload

ui:
	venv/bin/streamlit run frontend/app.py

eval:
	$(PYTHON) -m evaluation.eval_pipeline
