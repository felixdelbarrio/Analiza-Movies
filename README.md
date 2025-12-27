# 🎬 Analiza Movies

## 🇬🇧 English

**Analiza Movies** is a tool to analyze movie libraries (Plex / DLNA), compute quality scores
(IMDb / RT / Bayesian), suggest deletions or metadata fixes, and visualize results through
a modern Streamlit dashboard.

The project is designed with a **strict separation between backend and frontend**,
communicating exclusively through **disk artifacts (CSV / JSON)**.

### Key Features
- 🔍 Plex and DLNA movie library analysis
- 📊 Bayesian scoring with IMDb / Rotten Tomatoes
- 🧹 Detection of deletion candidates
- 🧠 Metadata correction suggestions
- 📈 Interactive Streamlit dashboard
- 🧩 Fully decoupled architecture (frontend ≠ backend)

### High-level Architecture
```
backend/        → produces data (CSV / JSON)
frontend/       → consumes data (UI)
data/           → persistent caches (JSON)
reports/        → final results (CSV)
```

- ❌ Frontend does NOT import backend
- ❌ No internal APIs
- ✅ File-based communication only
- ✅ Independent configuration per layer

---

## 🇪🇸 Español

**Analiza Movies** es una herramienta para analizar bibliotecas de películas (Plex / DLNA),
calcular scores de calidad (IMDb / RT / Bayes), sugerir borrados o correcciones de metadata
y visualizar los resultados mediante un dashboard moderno en Streamlit.

El proyecto está diseñado con una **separación estricta entre backend y frontend**,
comunicándose exclusivamente mediante **artefactos en disco (CSV / JSON)**.

### Características principales
- 🔍 Análisis de bibliotecas Plex y DLNA
- 📊 Scoring bayesiano con IMDb / Rotten Tomatoes
- 🧹 Detección de películas candidatas a borrado
- 🧠 Sugerencias de corrección de metadata
- 📈 Dashboard interactivo (Streamlit)
- 🧩 Arquitectura totalmente desacoplada (frontend ≠ backend)

### Arquitectura general
```
backend/        → genera datos (CSV / JSON)
frontend/       → consume datos (UI)
data/           → caches persistentes (JSON)
reports/        → resultados finales (CSV)
```

- ❌ El frontend NO importa backend
- ❌ No hay APIs internas
- ✅ Comunicación solo por ficheros
- ✅ Configuración independiente por capa

---

MIT License © Félix del Barrio
