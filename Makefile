SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
BLACK_VERSION := 25.12.0
NPM_CI := npm --prefix web ci --no-audit --fund=false

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
	@echo "make run       Reutiliza o genera el bundle nativo local y lo abre con su icono oficial"
	@echo "make build     Reutiliza o genera la distribución nativa para tu SO actual"
	@echo "make ci        Ejecuta build frontend + lint + black-check + typecheck + tests-cov"
	@echo "make test      Ejecuta pytest"
	@echo "make doctor    Diagnóstico del entorno"
	@echo "make reset     Borra el entorno y artefactos"
	@echo ""

$(PY):
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	@PIP_DISABLE_PIP_VERSION_CHECK=1 $(PY) -m pip install -q --upgrade pip setuptools wheel

$(PY_DEPS_STAMP): $(PY) requirements.txt requirements-dev.txt setup.py
	@PIP_DISABLE_PIP_VERSION_CHECK=1 $(PIP) install -q -r requirements-dev.txt
	@touch "$(PY_DEPS_STAMP)"

$(NODE_DEPS_STAMP): web/package-lock.json web/package.json
	@$(NPM_CI)
	@touch "$(NODE_DEPS_STAMP)"

$(WEB_BUILD_TARGET): $(WEB_SOURCES) $(NODE_DEPS_STAMP)
	@npm --prefix web run build

_install: $(PY_DEPS_STAMP) $(NODE_DEPS_STAMP)

install:
	@$(MAKE) reset
	@$(MAKE) _install

dev: install

run: $(PY_DEPS_STAMP) $(WEB_BUILD_TARGET)
	@echo "Lanzando bundle nativo local con branding oficial..."
	@$(PY) -m desktop.build --skip-frontend --no-archive --reuse-existing --quiet --run

build: $(PY_DEPS_STAMP) $(WEB_BUILD_TARGET)
	@echo "Empaquetando distribución nativa para $$(uname -s)..."
	@$(PY) -m desktop.build --skip-frontend --reuse-existing --quiet

ci: $(PY_DEPS_STAMP) $(WEB_BUILD_TARGET)
	@$(PY) -m ruff check .
	@$(PY) -m black --required-version $(BLACK_VERSION) --check .
	@$(PY) -m mypy src/backend src/desktop src/server src/shared
	@$(PY) -m pyright -p pyrightconfig.json
	@$(PY) -m pytest --cov=src --cov-branch

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
	@rm -f web/*.tsbuildinfo
	@rm -rf *.egg-info src/*.egg-info
	@rm -rf src/backend/logs
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name ".DS_Store" -delete
	@find . -type f \( -name ".coverage" -o -name ".coverage.*" \) -delete
	@find . -type d \( -name ".mypy_cache" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
	@rm -rf htmlcov

clean-venv:
	@rm -rf "$(VENV)"
