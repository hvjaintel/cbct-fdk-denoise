.PHONY: help setup stub-ir run pipeline webui docker-build docker-up docker-down clean

PYTHON ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
  PY := $(VENV)/Scripts/python.exe
  PIP := $(VENV)/Scripts/pip.exe
else
  PY := $(VENV)/bin/python
  PIP := $(VENV)/bin/pip
endif

help:
	@echo "CBCT-FDK-Denoise targets:"
	@echo "  make setup         Create venv + install Python deps"
	@echo "  make stub-ir       Generate OpenVINO stub denoiser IR"
	@echo "  make pipeline      Run CLI reconstruction (classical/auto)"
	@echo "  make webui         Build React dashboard"
	@echo "  make run           Generate stub IR (if needed) + serve dashboard"
	@echo "  make docker-build  Build container image"
	@echo "  make docker-up     Run via docker compose on :8080"
	@echo "  make docker-down   Stop compose stack"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

stub-ir:
	$(PY) scripts/generate_stub_ir.py

redcnn:
	$(PIP) install -r requirements-redcnn.txt
	$(PY) scripts/train_export_red_cnn.py

pipeline:
	$(PY) -m src.pipeline --volume-size 48 --angles 60 --detector 64 --denoise auto

pipeline-walnut:
	$(PY) -m src.pipeline --source walnut --walnut-dose 10 --volume-size 48 --angles 60 --detector 64 --denoise auto

walnut-10:
	$(PY) scripts/download_walnut.py --dose 10

walnut-all:
	$(PY) scripts/download_walnut.py --dose all

hku-hn:
	$(PIP) install -r requirements-hku.txt
	$(PY) scripts/download_hku_hn.py --case iba/hn/HN-CHUM-006

hku-hn-smoke:
	$(PIP) install -r requirements-hku.txt
	$(PY) scripts/download_hku_hn.py --case iba/hn/HN-CHUM-006 --max-files 60

webui:
	cd webapp && npm install && npm run build

run: stub-ir
	@if [ ! -d webapp/dist ]; then $(MAKE) webui; fi
	$(PY) backend/app.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	rm -rf output webapp/dist webapp/node_modules .venv
	rm -f models/denoise_stub/model.xml models/denoise_stub/model.bin models/denoise_stub/model.onnx
