# Makefile for Analiza-Movies
# Targets:
#   make backend
#   make run
#   make desktop
#   make frontend
#   make frontend-build
#   make build-local
#   make server
#   make doctor
#   make dev

SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

# ASGI module path
# You can override it like: make server-uvicorn API_MODULE=server.api.app:app
API_MODULE ?= server.api.app:app
API_HOST   ?= 0.0.0.0
API_PORT   ?= 8000

.DEFAULT_GOAL := help

.PHONY: help venv install dev reinstall run backend desktop build-local frontend frontend-install frontend-build frontend-preview \
        server server-uvicorn doctor typecheck mypy pyright lint format test test-cov clean clean-venv reset

help:
	@echo ""
	@echo "Analiza-Movies"
	@echo "--------------"
	@echo "make run             Ejecuta la app completa en contenedor nativo"
	@echo "make backend         Ejecuta el CLI backend (menú Plex/DLNA)"
	@echo "make desktop         Ejecuta la app de escritorio nativa"
	@echo "make frontend        Ejecuta el frontend React con Vite (solo desarrollo web)"
	@echo "make frontend-build  Genera el bundle de producción en web/dist"
	@echo "make frontend-preview Sirve localmente el build de React (solo QA web)"
	@echo "make build-local     Genera la distribución nativa para tu SO actual"
	@echo "make server          Ejecuta la API FastAPI (sirve web/dist si existe)"
	@echo "make server-uvicorn  Ejecuta la API FastAPI (via uvicorn: $(API_MODULE))"
	@echo ""
	@echo "make install         Crea venv e instala runtime (editable)"
	@echo "make dev             Instala runtime + tooling/dev deps"
	@echo "make reinstall       Reinstala el proyecto editable (runtime)"
	@echo "make frontend-install Instala dependencias Node del frontend"
	@echo "make typecheck       Ejecuta mypy y pyright (requiere make dev)"
	@echo "make mypy            Ejecuta solo mypy (requiere make dev)"
	@echo "make pyright         Ejecuta solo pyright (requiere make dev)"
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

# Dev install: incluye extras dev (mypy/pyright/stubs/black/ruff/pytest)
dev: venv
	@$(PIP) install -q -r requirements-dev.txt

reinstall: venv
	@$(PIP) install -q -r requirements-dev.txt

# -------------------------------------------------
# Targets principales
# -------------------------------------------------

run: desktop

backend: install
	@$(VENV)/bin/start

frontend-install:
	@npm --prefix web ci

frontend: frontend-install
	@npm --prefix web run dev -- --host 0.0.0.0

frontend-build: frontend-install
	@npm --prefix web run build

frontend-preview: frontend-build
	@npm --prefix web run preview -- --host 0.0.0.0

desktop: install frontend-build
	@$(VENV)/bin/start-desktop

build-local: dev frontend-build
	@echo "Empaquetando distribución nativa para $$(uname -s)..."
	@$(PY) -m desktop.build --skip-frontend

# Sirve la API y, si existe, también la SPA compilada en web/dist.
server: install
	@$(VENV)/bin/start-server

# Alternativa explícita para cuando el ASGI module cambie (p.ej., renombrado del fichero).
server-uvicorn: install
	@$(PY) -m uvicorn "$(API_MODULE)" --host "$(API_HOST)" --port "$(API_PORT)"

# -------------------------------------------------
# Calidad / Tipado (opcional)
# -------------------------------------------------

typecheck: dev
	@$(PY) -m mypy src/backend src/desktop src/server src/shared
	@$(PY) -m pyright .

mypy: dev
	@$(PY) -m mypy src/backend src/desktop src/server src/shared

pyright: dev
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
	@echo "Desktop cmd:   $$(test -x $(VENV)/bin/start-desktop && echo OK || echo MISSING)"
	@echo "ASGI module:   $(API_MODULE)"
	@echo "Type tools:    $$(($(PY) -c 'import mypy, pyright' >/dev/null 2>&1 && echo OK) || echo "(install with: make dev)")"
	@echo "Working dir:   $$(pwd)"

# -------------------------------------------------
# Limpieza
# -------------------------------------------------

clean:
	@rm -rf *.egg-info
	@rm -rf build dist dist-desktop
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name ".DS_Store" -delete

clean-venv:
	@rm -rf $(VENV)

reset: clean clean-venv
	@$(MAKE) dev
