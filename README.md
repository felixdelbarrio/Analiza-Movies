# 🎬 Analiza Movies

[![CI](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/ci.yml/badge.svg)](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/ci.yml)
[![Typing](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/typing.yml/badge.svg)](https://github.com/felixdelbarrio/Analiza-Movies/actions/workflows/typing.yml)
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
➡️ https://paypal.me/felixdelbarrio

---

## ⚡ Quickstart

1) Create environment files:
   - `cp .env.example .env`
   - `cp .env.front.example .env.front`
2) Install dependencies: `make dev`
3) Run:
   - Backend CLI: `make backend`
   - API server: `make server`
   - Frontend dashboard: `make frontend`

---

## 🖥️ Interactive Frontend (Streamlit)

The **frontend** is a first-class component of Analiza Movies, implemented using **Streamlit** and designed to sit directly on top of the generated reports and/or the REST API.

It provides a **human-in-the-loop** decision layer, turning raw analysis results into actionable insights.

### Frontend Capabilities

- 📑 **Tab-based navigation**, each tab focused on a specific analytical task  
- 🔎 **Advanced filtering and search** across large movie catalogs  
- 📊 **Charts, KPIs, and summary metrics**  
- 🧠 **Decision-oriented views** (KEEP / DELETE / REVIEW)  
- 🧾 **Metadata inspection and validation**  
- 🗑️ **Deletion candidate review** with safety-first logic  
- 🔌 **Dual data source support**:
  - CSV reports (offline / batch mode)
  - REST API (live / service mode)

### Main Dashboard Sections

- **All Movies**  
  Full catalog exploration with filters and sortable tables.

- **Candidates**  
  Automatically identified low-value or problematic titles.

- **Metadata**  
  Detection of mismatches, missing fields, and ambiguities.

- **Charts & Statistics**  
  Visual overview of quality distribution and trends.

- **Delete / Review**  
  Explicit human validation before any destructive action.

The frontend is intentionally **read-only by default**, ensuring that all deletion or corrective actions remain explicit and controlled.

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
➡️ https://paypal.me/felixdelbarrio

---

## ⚡ Inicio rápido

1) Crea los archivos de entorno:
   - `cp .env.example .env`
   - `cp .env.front.example .env.front`
2) Instala dependencias: `make dev`
3) Ejecuta:
   - Backend CLI: `make backend`
   - API server: `make server`
   - Frontend dashboard: `make frontend`

---

## 🖥️ Frontend Interactivo (Streamlit)

El **frontend** es un componente clave de Analiza Movies y actúa como la capa de **exploración, validación y decisión humana** sobre los resultados del análisis.

Está diseñado para trabajar tanto con **informes CSV** como con la **API REST**, permitiendo distintos modos de operación.

### Funcionalidades del Frontend

- Navegación clara por pestañas, orientada a tareas reales  
- Filtros avanzados para bibliotecas grandes  
- Métricas, gráficas y KPIs visuales  
- Vistas orientadas a decisión (KEEP / DELETE / REVIEW)  
- Revisión manual de candidatos antes de actuar  
- Consumo directo de API o ficheros generados  

### Casos de Uso Clave

- Explorar el catálogo completo  
- Identificar películas problemáticas  
- Validar candidatos a eliminación  
- Detectar errores de metadatos  
- Compartir resultados mediante CSV o HTML  

---

## 🧩 Main Components / Componentes Principales

- Backend Analyzer (CLI / batch)
- REST API Server (FastAPI)
- Interactive Frontend (Streamlit)
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
    Frontend[Streamlit Dashboard]
