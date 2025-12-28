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

.DEFAULT_GOAL := help

.PHONY: help venv install reinstall backend frontend server doctor clean clean-venv reset

help:
	@echo ""
	@echo "Analiza-Movies"
	@echo "--------------"
	@echo "make backend     Ejecuta el CLI backend (menú Plex/DLNA)"
	@echo "make frontend    Ejecuta el dashboard Streamlit"
	@echo "make server      Ejecuta la API FastAPI"
	@echo ""
	@echo "make doctor      Diagnóstico del entorno"
	@echo "make install     Crea venv e instala dependencias"
	@echo "make reinstall  Reinstala el proyecto editable"
	@echo "make reset       Borra venv y reinstala todo"
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

server: install
	@$(VENV)/bin/start-server

# -------------------------------------------------
# Diagnóstico
# -------------------------------------------------

doctor: venv
	@echo "Python:        $$($(PY) --version)"
	@echo "Python path:   $$($(PY) -c 'import sys; print(sys.executable)')"
	@echo "Pip:           $$($(PIP) -V)"
	@echo "Backend cmd:   $$(test -x $(VENV)/bin/start && echo OK || echo MISSING)"
	@echo "Server cmd:    $$(test -x $(VENV)/bin/start-server && echo OK || echo MISSING)"
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