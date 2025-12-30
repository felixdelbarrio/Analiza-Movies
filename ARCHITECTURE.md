---

# 📐 ARCHITECTURE.md

```markdown
# 📐 Analiza Movies – Architecture

> **EN / ES – Bilingual Technical Architecture**

---

## 🇬🇧 Architecture Overview (English)

Analiza Movies is designed as a **modular, resilient, and scalable media intelligence system**.

Its architecture cleanly separates **analysis**, **data access**, **API exposure**, and **visual exploration**, ensuring that each layer can evolve independently without cascading changes.

---

### Core Design Principles

- Strong modularity  
- Explicit separation of concerns  
- Configuration-driven behavior  
- Fault tolerance for external services  
- Deterministic and reproducible results  
- Observability and metrics  
- Long-term maintainability  

---

## 🇪🇸 Visión General de Arquitectura (Español)

Analiza Movies está diseñado como un **sistema de inteligencia multimedia modular, resiliente y escalable**.

La arquitectura separa claramente **análisis**, **acceso a datos**, **exposición por API** y **visualización**, permitiendo que cada capa evolucione de forma independiente.

---

### Principios de Diseño

- Modularidad extrema  
- Separación clara de responsabilidades  
- Comportamiento gobernado por configuración  
- Tolerancia a fallos en servicios externos  
- Resultados reproducibles  
- Observabilidad y métricas  
- Mantenibilidad a largo plazo  

---

## 🧩 System Layers

### 1️⃣ Backend Analysis Layer

**Responsibilities:**

- Media ingestion (Plex, DLNA, filesystem)  
- Normalization and canonicalization  
- Metadata enrichment (OMDb, Wikipedia)  
- Scoring and quality evaluation  
- Decision logic (keep / delete / review)  
- Report generation (CSV / HTML)  

This layer is **batch-oriented, deterministic, and side-effect controlled**.

---

### 2️⃣ Caching & Consolidation Layer

- Persistent OMDb and Wikipedia caches  
- File-based and HTTP caching  
- Consolidation of repeated lookups  
- Deterministic enrichment results  

This layer protects the system from:
- API rate limits  
- Network instability  
- Data drift  

---

### 3️⃣ API Layer (FastAPI)

The API layer exposes **read-only access** to:

- Consolidated movie data  
- Reports and metrics  
- Health and metadata endpoints  

Characteristics:

- Stateless  
- Typed and validated  
- Pagination and filtering  
- Designed for UI and automation  

---

### 4️⃣ Frontend Layer (Streamlit)

The frontend is a **consumer of analysis outputs**, never a producer of raw data.

**Key responsibilities:**

- Interactive exploration of large datasets  
- Visual analytics (charts, KPIs, summaries)  
- Human validation of automated decisions  
- Safe review of deletion candidates  
- Consumption of CSV or REST API data  

**Design choices:**

- Stateless UI logic  
- Explicit user intent for destructive actions  
- Clear separation between data, logic, and presentation  
- No hidden side effects  

The frontend acts as the **decision cockpit** of Analiza Movies.

---

## 🔄 End-to-End Analysis Flow

```mermaid
flowchart TD
    Input[Media Sources]
    Normalize[Normalization]
    Enrich[Metadata Enrichment]
    Score[Scoring Engine]
    Decide[Decision Logic]
    Report[Reports & Metrics]

    Input --> Normalize --> Enrich --> Score --> Decide --> Report