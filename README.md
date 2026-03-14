# 🎬 Analiza Movies

[![CI](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/ci.yml/badge.svg)](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/ci.yml)
[![Typing](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/typing.yml/badge.svg)](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/typing.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-2ea44f.svg)](https://github.com/sponsors/felixdelbarrio)
[![Donate](https://img.shields.io/badge/Donate-PayPal-blue.svg)](https://paypal.me/felixdelbarrio)

> **EN / ES – Bilingual Documentation**  
> Intelligent Media Library Analysis Platform

---

## 🇬🇧 Analiza Movies (English)

**Analiza Movies** is an advanced platform for analyzing, auditing, and optimizing multimedia libraries, designed for power users of **Plex**, **DLNA**, and external data sources such as **OMDb** and **Wikipedia**.

It combines **automated analysis**, **intelligent scoring**, **interactive dashboards**, and a **REST API** to help you make informed decisions about your collection: what to keep, improve, fix, or delete.

---

### 🚀 Value Proposition

- 📊 360° visibility of your media library  
- 🤖 Automated analysis with configurable scoring  
- 🧠 Metadata enrichment (OMDb + Wikipedia)  
- 🧹 Detection of duplicates, inconsistencies, and low-value content  
- 📈 Interactive dashboards and exportable reports  
- 🔌 REST API ready for integrations  
- 🛡️ Robust, modular, and scalable architecture  

---

### 💖 Support

If you find this project useful, you can support its development here:  
- GitHub Sponsors: https://github.com/sponsors/felixdelbarrio  
- PayPal: https://paypal.me/felixdelbarrio

---

## ⚡ Quickstart

1) Create environment files:
   - `cp .env.example .env`
   - `cp .env.front.example .env.front`
2) Install Python dependencies: `make dev`
3) Install frontend dependencies: `make frontend-install`
4) Run:
   - Backend CLI: `make backend`
   - API server: `make server`
   - React frontend (Vite dev): `make frontend`

### Production-like local run

1) Build the SPA: `make frontend-build`
2) Serve API + React bundle together: `make server`
3) Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🖥️ Interactive Frontend (React)

The frontend is now implemented from scratch in **React + TypeScript + Vite**.

It keeps the same analytical structure of the original dashboard, but moves to a component-based architecture that is significantly easier to evolve, optimize and brand at a higher visual level.

### Frontend Capabilities

- Cinematic navigation across dashboard, library, analytics, duplicates, metadata, cleanup and settings
- High-density exploration with virtualized tables and editorial detail panels
- ECharts-based visual storytelling with theme-aware rendering
- Profile-aware browsing across multiple Plex and DLNA origins
- React Query data loading on top of the FastAPI backend
- Safe delete execution through the API with dry-run support

### Development Model

- `make frontend` runs the Vite development server on port `5173`
- `make frontend-build` generates `web/dist`
- `make server` serves the FastAPI API and, if available, also serves `web/dist`

---

## 🇪🇸 Analiza Movies (Español)

**Analiza Movies** es una plataforma avanzada de análisis, auditoría y optimización de bibliotecas multimedia, diseñada para usuarios exigentes de **Plex**, **DLNA** y fuentes externas como **OMDb** y **Wikipedia**.

Combina **análisis automático**, **scoring inteligente**, **dashboards interactivos** y una **API REST** para ayudarte a decidir qué conservar, mejorar, corregir o eliminar.

---

### 🚀 Propuesta de Valor

- 📊 Visión 360° de tu biblioteca  
- 🤖 Análisis automático con scoring configurable  
- 🧠 Enriquecimiento de metadatos (OMDb + Wikipedia)  
- 🧹 Detección de duplicados y contenido de bajo valor  
- 📈 Dashboards interactivos y reportes exportables  
- 🔌 API REST lista para integraciones  
- 🛡️ Arquitectura robusta, modular y escalable  

---

### 💖 Apóyame

Si este proyecto te resulta útil, puedes apoyarlo aquí:  
- GitHub Sponsors: https://github.com/sponsors/felixdelbarrio  
- PayPal: https://paypal.me/felixdelbarrio

---

## ⚡ Inicio rápido

1) Crea los archivos de entorno:
   - `cp .env.example .env`
   - `cp .env.front.example .env.front`
2) Instala dependencias Python: `make dev`
3) Instala dependencias del frontend: `make frontend-install`
4) Ejecuta:
   - Backend CLI: `make backend`
   - API server: `make server`
   - Frontend React en desarrollo: `make frontend`

---

## 🖥️ Frontend Interactivo (React)

El frontend está ahora construido desde cero con **React + TypeScript + Vite**.

Mantiene la navegación, estructura analítica y riqueza visual del dashboard anterior, pero pasa a una arquitectura preparada para evolucionar con mucha más libertad en diseño, rendimiento y mantenibilidad.

### Funcionalidades del Frontend

- Navegación editorial entre dashboard, biblioteca, analítica, duplicados, metadata, limpieza y configuración
- Tablas virtualizadas para catálogos grandes
- Gráficos ECharts adaptados al tema visual activo
- Cambio global del origen visible entre múltiples perfiles Plex y DLNA
- Gestión de perfiles, descubrimiento en red y vinculación Plex desde la UI
- Acciones destructivas controladas vía API con soporte de simulación

### Ejecución

- `make frontend` arranca Vite para desarrollo
- `make frontend-build` genera `web/dist`
- `make server` sirve la API y, si existe el build, también sirve la SPA compilada

---

## 🧩 Main Components / Componentes Principales

- Backend Analyzer (CLI / batch)
- REST API Server (FastAPI)
- Interactive Frontend (React)
- Scoring & Decision Engine
- Caching & Resilience Layer
- Advanced Reporting

📐 **Architecture details / Detalle técnico:**  
➡️ [ARCHITECTURE.md](ARCHITECTURE.md)

## Typing support

This project is fully typed and compliant with **PEP 561**.
Type checkers such as **mypy** and **pyright** are fully supported.
![Typing: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational)
---

## 🏗️ High-Level Architecture

```mermaid
flowchart LR
    Plex[Plex Server]
    DLNA[DLNA Devices]
    OMDb[OMDb API]
    Wiki[Wikipedia API]

    Plex --> Backend
    DLNA --> Backend
    OMDb --> Backend
    Wiki --> Backend

    Backend --> Reports
    Backend --> API
    API --> Frontend

    Backend[Backend Analyzer]
    Reports[Reports & Metrics]
    API[FastAPI REST API]
    Frontend[React SPA]
```
