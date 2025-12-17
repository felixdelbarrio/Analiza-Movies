# Analiza Movies

> English version first · Versión en español a continuación

---

## English

### 1. Overview

Analiza Movies is a Python toolset to analyse a Plex movie library (and other DLNA-style sources) and decide which titles to **keep**, **delete** or **review**.

It combines:
- Metadata from your Plex server.
- External information from OMDb (IMDb, Rotten Tomatoes, etc.).
- Optional data from Wikipedia.
- Custom scoring and decision rules.
- A Streamlit dashboard to explore results and trigger deletions from Plex.

The project is designed to be **defensive**, **transparent** and **safe by default** (no physical deletion without explicit confirmation).

---

### 2. Main features

- Connects to a Plex server and scans one or more libraries.
- Fetches ratings and votes from OMDb, caching them on disk to avoid re-querying.
- Optionally enriches information using Wikipedia.
- Normalises all data into a single report (CSV) with one row per movie.
- Computes several scores (Bayesian and heuristic) to estimate quality and relevance.
- Applies explicit decision rules to classify each title as KEEP / DELETE / UNKNOWN / MISIDENTIFIED.
- Suggests metadata fixes for misidentified or low-quality entries (titles, years, collections, languages, etc.).
- Provides a Streamlit dashboard with:
  - Global statistics about the collection.
  - Detailed lists of all movies.
  - Focused views of deletion candidates.
  - Views for metadata issues and possible fixes.
- Integrates with Plex to perform physical deletion of selected movies (always gated by confirmation).
- Centralised logging with a configurable “silent mode” for unattended runs.

---

### 3. High-level architecture

At a high level the project is structured as a Python package (commonly named `backend`) with several coordinated modules:

- **Entry points**
  - `analiza.py`  
    Unified entry point that asks the user whether to analyse Plex or a DLNA source and delegates to the corresponding workflow.
  - `analiza_plex.py`  
    Main orchestrator for Plex analysis: connects to Plex, iterates over libraries, calls the analysis pipeline for each movie and produces the final report.

- **Configuration and logging**
  - `config.py`  
    Reads configuration from environment variables (via a `.env` file) and centralises all thresholds, flags and feature switches. Examples include:
    - Plex connection: `BASEURL`, `PLEX_TOKEN`.
    - OMDb access: `OMDB_API_KEY`, rate-limit behaviour and retry options.
    - Libraries to exclude: `EXCLUDE_LIBRARIES`.
    - Decision thresholds: e.g. minimum Rotten Tomatoes score, minimum IMDb rating, minimum number of votes, and similar values used by the scoring and decision logic.
  - `logger.py`  
    Thin wrapper around the standard logging module, honouring a `SILENT_MODE` configuration so that logs can be suppressed when required.

- **Plex and external services**
  - `plex_client.py`  
    Helpers to connect to the Plex server, retrieve libraries and movies, and perform deletion operations when requested.
  - `omdb_client.py`  
    Client for OMDb, including retry logic and a local on-disk cache (`omdb_cache.json`) of responses and extracted ratings.
  - `wiki_client.py`  
    Optional client for Wikipedia, also with local caching (`wiki_cache.json`) to avoid repeated HTTP calls.

- **Analysis and transformation**
  - `analyze_input_core.py` / `analyzer.py`  
    Core pipeline to transform the raw information about a single movie from Plex into a normalised “analysis row” that will later be consumed by reporting and dashboards. Handles defensive parsing, error handling and logging.
  - `metadata.py` and `metadata_fix.py`  
    Detection of potential metadata problems (misidentified movies, wrong years, missing or inconsistent fields) and generation of suggestions for fixing those issues.

- **Scoring and decision logic**
  - `scoring.py`  
    Implements Bayesian-style scoring and other auxiliary scores combining IMDb rating, Rotten Tomatoes score, number of votes, year, Plex user rating and similar signals.
  - `decision_logic.py`  
    Uses the scores and thresholds from `config.py` to assign each movie to one of several categories (KEEP, DELETE, UNKNOWN, MISIDENTIFIED, etc.) and to flag edge cases.
  - `delete_logic.py`  
    Encapsulates the rules and safety checks around physical deletion, ensuring that only explicitly confirmed candidates are deleted.

- **Reporting, statistics and dashboard**
  - `report_loader.py`  
    Helpers to load and validate the main report CSV (for example `report_all.csv`) and to convert it into a pandas DataFrame with the appropriate types.
  - `stats.py`  
    Functions to derive additional statistics from the report: distributions of ratings, votes, sizes, decades, languages and similar descriptive metrics.
  - `charts.py`  
    Chart-building helpers (based on Altair) used by the Streamlit dashboard.
  - `reporting.py` and `summary.py`  
    Build aggregated views and human-friendly summaries from the full analysis data.
  - `dashboard.py` and `components.py`  
    Implementation of the Streamlit dashboard: layout, pages, interactive controls and reusable UI components for listing movies, selecting candidates and drilling down into details.
  - `all_movies.py`, `candidates.py`, `delete.py`, `advanced.py`  
    Script or page-level modules that provide specific views or workflows within the dashboard, such as listing all titles, focusing on candidate deletions, performing deletion runs, or accessing advanced filters.

- **Auxiliary utilities and inputs**
  - `data_utils.py`  
    Shared helpers for working with pandas DataFrames and common transformations.
  - `dlna_input.py`  
    Abstractions for DLNA-style inputs when not using Plex directly.
  - `report_all.csv`  
    Example or existing full report produced by a previous analysis run.

---

### 4. Configuration

Configuration is provided primarily via environment variables, typically loaded from a `.env` file at startup. Some of the most important variables are:

- `BASEURL` – Base URL of the Plex server (including protocol and port).
- `PLEX_TOKEN` – Plex authentication token.
- `OMDB_API_KEY` – API key used to query OMDb.
- `EXCLUDE_LIBRARIES` – Comma-separated list of Plex libraries to skip.
- `OMDB_RATE_LIMIT_WAIT_SECONDS` – Waiting time when OMDb rate-limits requests.
- `OMDB_RATE_LIMIT_MAX_RETRIES` – Maximum number of retries after rate-limiting.
- `OMDB_RETRY_EMPTY_CACHE` – Whether to re-query OMDb when an entry is missing in the cache.
- Various decision thresholds for IMDb/Rotten Tomatoes ratings, minimum votes and similar criteria used by the scoring logic.

The defaults for these values and the full list of available options live in the configuration module.

---

### 5. Typical workflow

1. **Prepare configuration**
   - Create a `.env` file with at least the Plex and OMDb settings.
   - Optionally fine-tune thresholds and behaviour in your environment variables.

2. **Run a library analysis**
   - Execute the main entry point for analyses and select Plex (or DLNA) when prompted.
   - The tool connects to Plex, scans the libraries, fetches data from OMDb/Wikipedia as needed and writes an aggregated CSV report and caches.

3. **Inspect the dashboard**
   - Start the Streamlit dashboard that consumes the generated report.
   - Use the different pages to explore statistics, see all movies, inspect deletion candidates and review metadata issues and suggestions.

4. **Decide and delete (optional)**
   - Use the candidates and deletion views to mark titles to be removed.
   - Confirm deletions explicitly; only then are the files removed through the Plex API.
   - Review logs and summaries after completion.

---

### 6. Development notes

- The codebase uses type hints extensively and is intended to be friendly to static type checkers such as mypy or Pyright.
- Logging is centralised and configured lazily, making it safe to import modules without side effects.
- External HTTP calls are wrapped with retry and caching logic to minimise the impact of transient failures and API rate limits.
- The project also includes cached data files (`omdb_cache.json`, `wiki_cache.json`) and a sample report (`report_all.csv`) which can be useful during development or experimentation.

---

## Español

### 1. Descripción general

Analiza Movies es un conjunto de herramientas en Python para analizar una biblioteca de películas de Plex (y otras fuentes tipo DLNA) y decidir qué títulos **conservar**, **eliminar** o **revisar**.

Combina:

- Metadatos procedentes de tu servidor Plex.
- Información externa obtenida desde OMDb (IMDb, Rotten Tomatoes, etc.).
- Datos opcionales provenientes de Wikipedia.
- Reglas de puntuación y decisión personalizadas.
- Un panel interactivo en Streamlit para explorar resultados y lanzar eliminaciones en Plex.

El proyecto está pensado para ser **robusto**, **transparente** y **seguro por defecto** (no se borra nada físicamente sin una confirmación explícita).

---

### 2. Funcionalidades principales

- Se conecta a un servidor Plex y escanea una o varias bibliotecas.
- Obtiene votos y valoraciones desde OMDb y los guarda en caché en disco para evitar peticiones repetidas.
- Puede enriquecer la información utilizando Wikipedia de forma opcional.
- Normaliza todos los datos en un único informe (CSV) con una fila por película.
- Calcula varias puntuaciones (bayesianas y heurísticas) para estimar la calidad y relevancia de cada título.
- Aplica reglas de decisión claras para clasificar cada película como KEEP / DELETE / UNKNOWN / MISIDENTIFIED.
- Genera sugerencias de corrección de metadatos para entradas mal identificadas o de baja calidad (títulos, años, colecciones, idiomas, etc.).
- Ofrece un panel en Streamlit con:
  - Estadísticas globales de la colección.
  - Listados detallados de todas las películas.
  - Vistas centradas en candidatas a borrar.
  - Vistas de problemas de metadatos y posibles soluciones.
- Se integra con Plex para realizar el borrado físico de películas seleccionadas (siempre protegido por confirmación).
- Usa un sistema de logging centralizado con un modo silencioso configurable para ejecuciones desatendidas.

---

### 3. Arquitectura a alto nivel

A alto nivel, el proyecto se organiza como un paquete de Python (habitualmente llamado `backend`) con varios módulos coordinados:

- **Puntos de entrada**
  - `analiza.py`  
    Punto de entrada unificado que pregunta al usuario si quiere analizar Plex o una fuente DLNA y delega en el flujo correspondiente.
  - `analiza_plex.py`  
    Orquestador principal para el análisis de Plex: se conecta al servidor, recorre las bibliotecas, llama al pipeline de análisis para cada película y produce el informe final.

- **Configuración y logging**
  - `config.py`  
    Lee la configuración desde variables de entorno (cargadas desde un fichero `.env`) y centraliza todos los umbrales, flags y opciones. Entre otros:
    - Conexión a Plex: `BASEURL`, `PLEX_TOKEN`.
    - Acceso a OMDb: `OMDB_API_KEY`, parámetros de reintento y limitación de uso.
    - Bibliotecas a excluir: `EXCLUDE_LIBRARIES`.
    - Umbrales de decisión: puntuaciones mínimas de Rotten Tomatoes, rating mínimo de IMDb, votos mínimos, etc.
  - `logger.py`  
    Capa fina sobre el módulo estándar de logging, respetando el ajuste `SILENT_MODE` para desactivar logs cuando sea necesario.

- **Plex y servicios externos**
  - `plex_client.py`  
    Funciones auxiliares para conectarse a Plex, obtener bibliotecas y películas y ejecutar borrados cuando se confirman.
  - `omdb_client.py`  
    Cliente para OMDb con lógica de reintentos y un sistema de caché en disco (`omdb_cache.json`) para las respuestas y valoraciones.
  - `wiki_client.py`  
    Cliente opcional para Wikipedia, con caché local (`wiki_cache.json`) para reducir el número de solicitudes HTTP.

- **Análisis y transformación**
  - `analyze_input_core.py` / `analyzer.py`  
    Núcleo del pipeline que transforma la información cruda de una película en Plex en una “fila de análisis” normalizada que después consumen los módulos de reporting y el dashboard. Incluye manejo defensivo de errores y logging.
  - `metadata.py` y `metadata_fix.py`  
    Detección de problemas de metadatos (películas mal identificadas, años incorrectos, campos ausentes o incoherentes) y generación de sugerencias de corrección.

- **Puntuación y lógica de decisión**
  - `scoring.py`  
    Implementa puntuaciones de tipo bayesiano y otras métricas auxiliares combinando rating de IMDb, puntuación de Rotten Tomatoes, número de votos, año de estreno, rating de usuario en Plex, etc.
  - `decision_logic.py`  
    Usa las puntuaciones y los umbrales definidos en `config.py` para asignar cada película a categorías como KEEP, DELETE, UNKNOWN, MISIDENTIFIED y para marcar casos borde.
  - `delete_logic.py`  
    Encapsula las reglas y comprobaciones de seguridad relacionadas con el borrado físico, garantizando que solo se eliminan títulos cuando el usuario lo confirma expresamente.

- **Informes, estadísticas y panel**
  - `report_loader.py`  
    Funciones para cargar y validar el informe principal en CSV (por ejemplo `report_all.csv`) y convertirlo en un DataFrame de pandas con tipos adecuados.
  - `stats.py`  
    Cálculo de estadísticas derivadas: distribuciones de ratings, votos, tamaños, décadas, idiomas y otras métricas descriptivas.
  - `charts.py`  
    Constructores de gráficas (basados en Altair) utilizados por el panel en Streamlit.
  - `reporting.py` y `summary.py`  
    Construyen vistas agregadas y resúmenes legibles a partir de los datos de análisis.
  - `dashboard.py` y `components.py`  
    Implementación del panel en Streamlit: estructura de páginas, controles interactivos y componentes reutilizables para listar películas, seleccionar candidatas y profundizar en los detalles.
  - `all_movies.py`, `candidates.py`, `delete.py`, `advanced.py`  
    Módulos de página o scripts que proporcionan vistas o flujos concretos dentro del panel: listado de todas las películas, candidata a eliminación, ejecuciones de borrado y filtros avanzados.

- **Utilidades auxiliares e inputs**
  - `data_utils.py`  
    Utilidades compartidas para trabajar con DataFrames de pandas y transformaciones comunes.
  - `dlna_input.py`  
    Abstracción para entradas tipo DLNA cuando no se usa Plex directamente.
  - `report_all.csv`  
    Informe completo de ejemplo o generado en una ejecución anterior del análisis.

---

### 4. Configuración

La configuración se define principalmente a través de variables de entorno, normalmente cargadas desde un fichero `.env` al inicio de la ejecución. Algunas de las más importantes son:

- `BASEURL` – URL base del servidor Plex (incluyendo protocolo y puerto).
- `PLEX_TOKEN` – Token de autenticación de Plex.
- `OMDB_API_KEY` – Clave de API utilizada para consultar OMDb.
- `EXCLUDE_LIBRARIES` – Lista separada por comas de bibliotecas Plex a excluir del análisis.
- `OMDB_RATE_LIMIT_WAIT_SECONDS` – Tiempo de espera cuando OMDb aplica limitación de uso.
- `OMDB_RATE_LIMIT_MAX_RETRIES` – Número máximo de reintentos tras una limitación.
- `OMDB_RETRY_EMPTY_CACHE` – Indica si se debe reconsultar OMDb cuando falte una entrada en la caché.
- Varios umbrales de decisión para ratings de IMDb/Rotten Tomatoes, número mínimo de votos, etc., utilizados por la lógica de puntuación.

Los valores por defecto y la lista completa de opciones se encuentran en el módulo de configuración.

---

### 5. Flujo de trabajo típico

1. **Preparar la configuración**
   - Crear un fichero `.env` con, como mínimo, los datos de conexión a Plex y la clave de OMDb.
   - Ajustar si se desea los umbrales y el comportamiento mediante variables de entorno.

2. **Ejecutar el análisis de la biblioteca**
   - Lanzar el punto de entrada principal y elegir Plex (o DLNA) cuando se solicite.
   - La herramienta se conecta a Plex, recorre las bibliotecas, consulta OMDb/Wikipedia según sea necesario y genera un informe CSV agregado y las cachés correspondientes.

3. **Revisar el panel**
   - Iniciar el panel en Streamlit que consume el informe generado.
   - Utilizar sus distintas páginas para ver estadísticas, consultar el listado completo, inspeccionar candidatas a borrado y revisar problemas de metadatos y sugerencias.

4. **Decidir y borrar (opcional)**
   - Desde las vistas de candidatas y borrado, marcar los títulos que se desea eliminar.
   - Confirmar de forma explícita las eliminaciones; solo entonces se borran los archivos mediante la API de Plex.
   - Revisar los logs y los resúmenes tras la ejecución.

---

### 6. Notas de desarrollo

- El código utiliza anotaciones de tipos de manera intensiva y está pensado para funcionar bien con analizadores estáticos como mypy o Pyright.
- El sistema de logging está centralizado y se configura de forma perezosa, por lo que es seguro importar módulos sin provocar efectos secundarios.
- Las llamadas HTTP a servicios externos están envueltas en lógica de reintentos y caché para reducir el impacto de errores transitorios y limitaciones de uso.
- El repositorio incluye ficheros de caché (`omdb_cache.json`, `wiki_cache.json`) y un informe de ejemplo (`report_all.csv`), útiles durante el desarrollo o la experimentación.