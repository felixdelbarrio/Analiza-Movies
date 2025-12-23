# Analiza Movies

## 🇬🇧 English

### What is Analiza Movies?
**Analiza Movies** is a professional-grade tool designed to analyze, score, and clean large movie libraries from **Plex** and **DLNA** servers.  
It enriches your collection with external metadata (IMDb, Rotten Tomatoes, Wikipedia), applies customizable scoring rules, and helps you **decide what to keep, delete, or review** — all backed by reports and a modern dashboard.

### Key Features
- 🎬 Plex & DLNA library ingestion
- ⭐ IMDb / Rotten Tomatoes scoring
- 🧠 Intelligent decision engine (KEEP / DELETE / MAYBE)
- 🗂 Metadata normalization and fixing
- 📊 CSV & HTML reports
- 🖥 Interactive Streamlit dashboard
- ⚡ Local cache for OMDb & Wikipedia (fast & API-friendly)

### Requirements
- Python **3.10+**
- Plex server (optional)
- OMDb API key

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Configuration
Create a `.env` file in the project root:
```env
OMDB_API_KEY=your_key_here
PLEX_URL=http://localhost:32400
PLEX_TOKEN=your_token_here
```

### Running the CLI
```bash
analiza
analiza-plex
analiza-dlna
```

### Launching the Dashboard
```bash
streamlit run dashboard.py
```

### Outputs
- `Reports/report_all.csv`
- `Reports/report_filtered.csv`
- `Reports/metadata_fix.csv`
- `frontend/templates/filtered_report.html`

---

## 🇪🇸 Español

### ¿Qué es Analiza Movies?
**Analiza Movies** es una herramienta profesional para analizar, puntuar y depurar grandes bibliotecas de películas en **Plex** y **DLNA**.  
Enriquece tu colección con metadatos externos (IMDb, Rotten Tomatoes, Wikipedia) y aplica reglas inteligentes para ayudarte a **decidir qué conservar, borrar o revisar**.

### Características principales
- 🎬 Ingesta de librerías Plex y DLNA
- ⭐ Puntuaciones IMDb / Rotten Tomatoes
- 🧠 Motor de decisiones (KEEP / DELETE / MAYBE)
- 🗂 Corrección y normalización de metadatos
- 📊 Informes CSV y HTML
- 🖥 Dashboard interactivo con Streamlit
- ⚡ Caché local para APIs externas

### Requisitos
- Python **3.10+**
- Servidor Plex (opcional)
- Clave de API de OMDb

### Instalación
```bash
pip install -r requirements.txt
pip install -e .
```

### Configuración
Crea un archivo `.env`:
```env
OMDB_API_KEY=tu_api_key
PLEX_URL=http://localhost:32400
PLEX_TOKEN=tu_token
```

### Ejecución por CLI
```bash
analiza
analiza-plex
analiza-dlna
```

### Lanzar el Dashboard
```bash
streamlit run dashboard.py
```

### Resultados
- `Reports/report_all.csv`
- `Reports/report_filtered.csv`
- `Reports/metadata_fix.csv`
- `frontend/templates/filtered_report.html`
