# Makefile for Analiza-Movies
# Targets:
#   make backend
#   make frontend
#   make server
#   make doctor

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

.PHONY: help venv install reinstall backend frontend server server-uvicorn doctor clean clean-venv reset

help:
	@echo ""
	@echo "Analiza-Movies"
	@echo "--------------"
	@echo "make backend         Ejecuta el CLI backend (menú Plex/DLNA)"
	@echo "make frontend        Ejecuta el dashboard Streamlit"
	@echo "make server          Ejecuta la API FastAPI (via start-server)"
	@echo "make server-uvicorn  Ejecuta la API FastAPI (via uvicorn: $(API_MODULE))"
	@echo ""
	@echo "make doctor          Diagnóstico del entorno"
	@echo "make install         Crea venv e instala dependencias"
	@echo "make reinstall       Reinstala el proyecto editable"
	@echo "make reset           Borra venv y reinstala todo"
	@echo ""

# -------------------------------------------------
# Entorno
# -------------------------------------------------

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	@$(PY) -m pip install -q --upgrade pip

install: venv
	@$(PIP) install -q -r requirements.txt
	@$(PIP) install -q -e .

reinstall: venv
	@$(PIP) install -q -e .

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
# No reemplaza a `server` para no romper el flujo actual.
server-uvicorn: install
	@$(PY) -m uvicorn "$(API_MODULE)" --host "$(API_HOST)" --port "$(API_PORT)"

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
	@echo "Working dir:   $$(pwd)"

# -------------------------------------------------
# Limpieza
# -------------------------------------------------

clean:
	@rm -rf *.egg-info
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name ".DS_Store" -delete

clean-venv:
	@rm -rf $(VENV)

reset: clean clean-venv
	@$(MAKE) install