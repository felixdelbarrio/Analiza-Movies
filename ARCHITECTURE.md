# 🏗️ Analiza Movies Architecture

## 🇬🇧 English

### Core Goal
Guarantee that:
- Backend is a **data-producing engine**
- Frontend is a **passive consumer**
- Both can evolve **independently**

### Core Principles
| Layer | Can import | Must NOT import |
|------|------------|-----------------|
| Backend | backend.*, std | frontend.* |
| Frontend | frontend.*, std | backend.* ❌ |

### Data Flow
```
backend → CSV / JSON → frontend
```

### Backend Responsibilities
- Analyze Plex / DLNA
- Query OMDb / Wikipedia
- Persist caches in `data/`
- Generate reports in `reports/`
- No UI logic

### Frontend Responsibilities
- Read CSV / JSON only
- Compute lightweight metrics
- Render UI (Streamlit)
- No business logic

### Configuration
- Backend: `.env`
- Frontend: `.env.front`
- No fallback between them

---

## 🇪🇸 Español

### Objetivo principal
Garantizar que:
- El backend sea un **motor productor de datos**
- El frontend sea un **consumidor pasivo**
- Ambos puedan evolucionar **de forma independiente**

### Principios clave
| Capa | Puede importar | NO puede importar |
|-----|---------------|------------------|
| Backend | backend.*, std | frontend.* |
| Frontend | frontend.*, std | backend.* ❌ |

### Flujo de datos
```
backend → CSV / JSON → frontend
```

### Responsabilidades del Backend
- Analizar Plex / DLNA
- Consultar OMDb / Wikipedia
- Persistir caches en `data/`
- Generar reports en `reports/`
- Sin lógica de UI

### Responsabilidades del Frontend
- Leer solo CSV / JSON
- Calcular métricas ligeras
- Renderizar UI (Streamlit)
- Sin lógica de negocio

### Configuración
- Backend: `.env`
- Frontend: `.env.front`
- Sin fallback entre ellos

---

This architecture minimizes coupling, simplifies debugging, and enables future scaling.
