# Makefile for Analiza-Movies
# Targets:
#   make backend
#   make frontend
#   make server
#   make doctor
#   make dev

SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

# ASGI module path (updated after renaming api_fastapi.py -> am_api.py)
# You can override it like: make server-uvicorn API_MODULE=am_api:app
API_MODULE ?= am_api:app
API_HOST   ?= 0.0.0.0
API_PORT   ?= 8000

.DEFAULT_GOAL := help

.PHONY: help venv install dev reinstall backend frontend server server-uvicorn doctor \
        typecheck lint format test test-cov clean clean-venv reset

help:
	@echo ""
	@echo "Analiza-Movies"
	@echo "--------------"
	@echo "make backend         Ejecuta el CLI backend (menú Plex/DLNA)"
	@echo "make frontend        Ejecuta el dashboard Streamlit"
	@echo "make server          Ejecuta la API FastAPI (via start-server)"
	@echo "make server-uvicorn  Ejecuta la API FastAPI (via uvicorn: $(API_MODULE))"
	@echo ""
	@echo "make install         Crea venv e instala runtime (editable)"
	@echo "make dev             Instala runtime + tooling/dev deps"
	@echo "make reinstall       Reinstala el proyecto editable (runtime)"
	@echo "make typecheck       Ejecuta mypy y pyright (requiere make dev)"
	@echo "make mypy            Ejecuta solo mypy (requiere make dev)"
	@echo "make clean-streamlit Limpia caches de componentes de Streamlit"
	@echo "make lint            Ejecuta ruff (requiere make dev)"
	@echo "make format          Ejecuta black + ruff format (requiere make dev)"
	@echo "make test            Ejecuta pytest (requiere make dev)"
	@echo "make test-cov        Ejecuta pytest con cobertura (requiere make dev)"
	@echo ""
	@echo "make doctor          Diagnóstico del entorno"
	@echo "make reset           Borra venv y reinstala todo (dev)"
	@echo ""

# -------------------------------------------------
# Entorno
# -------------------------------------------------

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	@$(PY) -m pip install -q --upgrade pip setuptools wheel

# Runtime install: setup.py / install_requires es la fuente de verdad
install: venv
	@$(PIP) install -q -r requirements.txt
	@test ! -f requirements-png.txt || $(PIP) install -q -r requirements-png.txt

# Dev install: incluye extras dev (mypy/pyright/stubs/black/ruff/pytest)
dev: venv
	@$(PIP) install -q -r requirements-dev.txt
	@test ! -f requirements-png.txt || $(PIP) install -q -r requirements-png.txt

reinstall: venv
	@$(PIP) install -q -r requirements-dev.txt
	@test ! -f requirements-png.txt || $(PIP) install -q -r requirements-png.txt

# -------------------------------------------------
# Targets principales
# -------------------------------------------------

backend: install
	@$(VENV)/bin/start

frontend: install
	@$(VENV)/bin/start --dashboard

# Mantiene el comportamiento actual: usa el entrypoint instalado.
server: install
	@$(VENV)/bin/start-server

# Alternativa explícita para cuando el ASGI module cambie (p.ej., renombrado del fichero).
server-uvicorn: install
	@$(PY) -m uvicorn "$(API_MODULE)" --host "$(API_HOST)" --port "$(API_PORT)"

# -------------------------------------------------
# Calidad / Tipado (opcional)
# -------------------------------------------------

typecheck: dev
	@$(PY) -m mypy .

mypy: dev
	@$(PY) -m mypy .
	@$(PY) -m pyright .

lint: dev
	@$(PY) -m ruff check .

format: dev
	@$(PY) -m black .
	@$(PY) -m ruff format .

test: dev
	@$(PY) -m pytest

test-cov: dev
	@$(PY) -m pytest --cov=. --cov-branch

# -------------------------------------------------
# Diagnóstico
# -------------------------------------------------

doctor: venv
	@echo "Python:        $$($(PY) --version)"
	@echo "Python path:   $$($(PY) -c 'import sys; print(sys.executable)')"
	@echo "Pip:           $$($(PIP) -V)"
	@echo "Backend cmd:   $$(test -x $(VENV)/bin/start && echo OK || echo MISSING)"
	@echo "Server cmd:    $$(test -x $(VENV)/bin/start-server && echo OK || echo MISSING)"
	@echo "ASGI module:   $(API_MODULE)"
	@echo "Type tools:    $$(($(PY) -c 'import mypy, pyright' >/dev/null 2>&1 && echo OK) || echo "(install with: make dev)")"
	@echo "Working dir:   $$(pwd)"

# -------------------------------------------------
# Limpieza
# -------------------------------------------------

clean:
	@rm -rf *.egg-info
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name ".DS_Store" -delete

clean-streamlit:
	@rm -rf ~/.streamlit/components

clean-venv:
	@rm -rf $(VENV)

reset: clean clean-venv
	@$(MAKE) dev
