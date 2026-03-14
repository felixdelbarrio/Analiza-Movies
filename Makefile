SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

PY_DEPS_STAMP := $(VENV)/.deps-ready
NODE_DEPS_STAMP := web/node_modules/.deps-ready
WEB_BUILD_TARGET := web/dist/index.html
WEB_SOURCES := $(shell find web/src -type f) \
	web/package.json \
	web/package-lock.json \
	web/index.html \
	web/tsconfig.json \
	web/tsconfig.app.json \
	web/tsconfig.node.json \
	web/vite.config.ts

.DEFAULT_GOAL := help

.PHONY: help install dev run build ci test doctor reset clean clean-venv _install

help:
	@echo ""
	@echo "Analiza-Movies"
	@echo "--------------"
	@echo "make install   Reinicia el entorno e instala dependencias"
	@echo "make dev       Alias de make install"
	@echo "make run       Ejecuta la app completa en contenedor nativo"
	@echo "make build     Genera la distribución nativa para tu SO actual"
	@echo "make ci        Ejecuta build frontend + lint + black-check + typecheck + tests-cov"
	@echo "make test      Ejecuta pytest"
	@echo "make doctor    Diagnóstico del entorno"
	@echo "make reset     Borra el entorno y artefactos"
	@echo ""

$(PY):
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	@$(PY) -m pip install -q --upgrade pip setuptools wheel

$(PY_DEPS_STAMP): $(PY) requirements.txt requirements-dev.txt setup.py
	@$(PIP) install -q -r requirements-dev.txt
	@touch "$(PY_DEPS_STAMP)"

$(NODE_DEPS_STAMP): web/package-lock.json web/package.json
	@npm --prefix web ci
	@touch "$(NODE_DEPS_STAMP)"

$(WEB_BUILD_TARGET): $(WEB_SOURCES) $(NODE_DEPS_STAMP)
	@npm --prefix web run build

_install: $(PY_DEPS_STAMP) $(NODE_DEPS_STAMP)

install:
	@$(MAKE) reset
	@$(MAKE) _install

dev: install

run: $(PY_DEPS_STAMP) $(WEB_BUILD_TARGET)
	@$(VENV)/bin/start-desktop

build: $(PY_DEPS_STAMP) $(WEB_BUILD_TARGET)
	@echo "Empaquetando distribución nativa para $$(uname -s)..."
	@$(PY) -m desktop.build --skip-frontend

ci: $(PY_DEPS_STAMP) $(WEB_BUILD_TARGET)
	@$(PY) -m ruff check .
	@$(PY) -m black --check .
	@$(PY) -m mypy src/backend src/desktop src/server src/shared
	@$(PY) -m pyright -p pyrightconfig.json
	@$(PY) -m pytest --cov=. --cov-branch

test: $(PY_DEPS_STAMP)
	@$(PY) -m pytest

doctor: $(PY)
	@echo "Python:        $$($(PY) --version)"
	@echo "Python path:   $$($(PY) -c 'import sys; print(sys.executable)')"
	@echo "Pip:           $$($(PIP) -V)"
	@echo "Node:          $$(node --version 2>/dev/null || echo MISSING)"
	@echo "NPM:           $$(npm --version 2>/dev/null || echo MISSING)"
	@echo "Desktop cmd:   $$(test -x $(VENV)/bin/start-desktop && echo OK || echo MISSING)"
	@echo "Working dir:   $$(pwd)"

reset: clean clean-venv

clean:
	@rm -rf build dist dist-desktop
	@rm -rf web/dist web/node_modules
	@rm -rf *.egg-info src/*.egg-info
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name ".DS_Store" -delete
	@find . -type d \( -name ".mypy_cache" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +

clean-venv:
	@rm -rf "$(VENV)"
