from __future__ import annotations

"""
backend/analiza_omdb.py

Orquestador de enriquecimiento OMDb (sin UI).

Responsabilidades
-----------------
- Invocar el cliente OMDb/cache (backend.omdb_client) de forma robusta (best-effort).
- Normalizar/extraer campos útiles (ratings/votes/rt) desde el payload OMDb.
- Exponer una estructura “digerida” para el resto del pipeline.
- (Opcional) Persistir "parches" a records cacheados vía patch_cached_omdb_record
  para evitar recalcular/reenriquecer en futuras ejecuciones.

No-responsabilidades (delegadas a omdb_client.py)
------------------------------------------------
- HTTP, retries, throttle, semaphore, single-flight, circuit breaker.
- Cache persistente, compaction, TTL, negative caching.
- Candidate search (s=) y fallback a i=.
- Política de deshabilitado (OMDB_DISABLED) y manejo de API key inválida.

Diseño
------
- Este módulo NO decide si OMDb está "enabled": lo respeta vía omdb_client (OMDB_DISABLED, API key, etc.).
- Logs centralizados usando backend/logger.py:
    - logger.debug_ctx("OMDB", ...) para diagnóstico (gated por DEBUG_MODE).
    - logger.warning(..., always=True) sólo si fuera estrictamente necesario (aquí normalmente no).
- Fail-safe: cualquier excepción aquí NO debe romper el run; devolvemos None/valores vacíos.

Ajustes alineados con omdb_client.py (schema v4)
------------------------------------------------
1) Empty-ratings awareness:
   - OMDb puede devolver Response=True pero sin ratings (cache status "empty_ratings").
   - El pipeline debe distinguir "enrichment válido sin ratings" para evitar reintentos o confusiones.
   - Se expone:
       - OmdbEnrichment.has_ratings
       - metadata["omdb_has_ratings"] (si se aplica)

2) Normalización defensiva de Year:
   - OMDb puede devolver Year como "2019–2020", "2019-2020", etc.
   - Canonizamos a los 4 primeros dígitos si existen.

3) Writeback robusto:
   - Normalizamos imdb_id (lower/strip) antes de patch_cached_omdb_record.
   - Recomendación de uso: si dispones de OmdbEnrichment, usa enrichment.imdb_id (confirmado).

BONUS) Diagnóstico suave en Response=False:
   - Si OMDb responde Response=False, devolvemos None (sin enrichment),
     pero dejamos traza en DEBUG con el "Error" si existe.

Configuración
-------------
- Este módulo no introduce configs nuevas. Toda la política (TTL, throttling, métricas, etc.)
  vive en backend/config_omdb.py y backend/omdb_client.py.

API pública
-----------
- enrich_with_omdb(title, year, imdb_id, provenance=None) -> OmdbEnrichment | None
- apply_omdb_enrichment_to_metadata(metadata, enrichment, overwrite=False) -> dict (in-place shallow)
- writeback_omdb_wiki_block(norm_title, norm_year, imdb_id, wiki_block) -> bool
"""

from dataclasses import dataclass
from typing import Mapping

from backend import logger
from backend.movie_input import normalize_title_for_lookup
from backend.omdb_client import (
    extract_ratings_from_omdb,
    omdb_query_with_cache,
    patch_cached_omdb_record,
)

# =============================================================================
# Logging (centralizado)
# =============================================================================


def _dbg(msg: object) -> None:
    """Diagnóstico contextual (solo si DEBUG_MODE=True)."""
    logger.debug_ctx("OMDB", msg)


# =============================================================================
# Data model (lo que el resto del pipeline realmente necesita)
# =============================================================================


@dataclass(frozen=True)
class OmdbEnrichment:
    """
    Resultado “digerido” para el pipeline.

    - payload: dict OMDb completo (para consumidores avanzados)
    - imdb_rating, imdb_votes, rt_score: valores ya parseados (None si no disponibles)
    - has_ratings: True si existe al menos uno de los 3 (rating/votes/rt)
    - imdb_id: imdbID final (si OMDb lo devuelve)
    - norm_title, norm_year: canon de cache/pipeline (normalizados)
    """

    payload: dict[str, object]
    imdb_rating: float | None
    imdb_votes: int | None
    rt_score: int | None
    has_ratings: bool
    imdb_id: str | None
    norm_title: str
    norm_year: str


# =============================================================================
# Core orchestrator
# =============================================================================


def enrich_with_omdb(
    *,
    title: str | None,
    year: int | None,
    imdb_id: str | None,
    provenance: Mapping[str, object] | None = None,
) -> OmdbEnrichment | None:
    """
    Obtiene OMDb (cache-first) y devuelve un objeto OmdbEnrichment.

    Best-effort:
    - Si falla OMDb o no hay datos utilizables, devuelve None.
    - No lanza excepciones al caller.

    Nota:
    - Response=False => None (sin enrichment), pero en DEBUG loguea el error (BONUS).
    - Response=True pero sin ratings => enrichment válido con has_ratings=False.
    """
    try:
        norm_title = normalize_title_for_lookup(title or "") or ""
        norm_year = str(year) if year is not None else ""

        imdb_hint = (imdb_id or "").strip()
        if not norm_title and not imdb_hint:
            return None

        data = omdb_query_with_cache(
            title=title,
            year=year,
            imdb_id=imdb_id,
            provenance=provenance,
        )
        if not isinstance(data, dict):
            return None

        # Response=False => sin enrichment (pero log suave en DEBUG)
        if data.get("Response") != "True":
            err = data.get("Error")
            if err is not None:
                _dbg(f"OMDb Response=False: {err!r}")
            return None

        imdb_rating, imdb_votes, rt_score = extract_ratings_from_omdb(data)
        has_ratings = not (imdb_rating is None and imdb_votes is None and rt_score is None)

        imdb_id_final = data.get("imdbID")
        if not isinstance(imdb_id_final, str) or not imdb_id_final.strip():
            imdb_id_final = None
        else:
            imdb_id_final = imdb_id_final.strip().lower()

        # title/year canónicos (si OMDb devolvió mejores)
        t = data.get("Title")
        if isinstance(t, str) and t.strip():
            norm_title = normalize_title_for_lookup(t) or norm_title

        # Year defensivo: "2019–2020", "2019-2020", etc. -> "2019"
        y = data.get("Year")
        if isinstance(y, str):
            y4 = y.strip()[:4]
            if y4.isdigit():
                norm_year = y4

        return OmdbEnrichment(
            payload=dict(data),
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rt_score=rt_score,
            has_ratings=has_ratings,
            imdb_id=imdb_id_final,
            norm_title=norm_title,
            norm_year=norm_year,
        )

    except Exception as exc:
        _dbg(f"enrich_with_omdb failed: {exc!r}")
        return None


# =============================================================================
# Apply to pipeline metadata
# =============================================================================


def apply_omdb_enrichment_to_metadata(
    metadata: dict[str, object],
    enrichment: OmdbEnrichment,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """
    Aplica campos “útiles” al dict metadata (shallow).

    Regla:
    - Si overwrite=False, no pisa valores existentes no vacíos.

    Campos aplicados:
    - omdb_imdb_rating, omdb_imdb_votes, omdb_rt_score
    - omdb_has_ratings (bool): indica si hay al menos uno de esos campos
    - imdbID (opcional): imdb_id confirmado (si existe)
    """
    try:

        def put(key: str, value: object) -> None:
            if value is None:
                return
            if not overwrite:
                existing = metadata.get(key)
                if existing not in (None, "", "N/A"):
                    return
            metadata[key] = value

        put("omdb_imdb_rating", enrichment.imdb_rating)
        put("omdb_imdb_votes", enrichment.imdb_votes)
        put("omdb_rt_score", enrichment.rt_score)

        # Siempre útil, y no depende de ratings concretos.
        put("omdb_has_ratings", bool(enrichment.has_ratings))

        # opcional: guardar imdb_id “confirmado”
        put("imdbID", enrichment.imdb_id)

        # opcional: guardar payload completo (trazabilidad)
        # OJO: puede crecer; dejar desactivado por defecto.
        # put("omdb_payload", enrichment.payload)

        return metadata

    except Exception as exc:
        _dbg(f"apply_omdb_enrichment_to_metadata failed: {exc!r}")
        return metadata


# =============================================================================
# Cache write-back helpers (p.ej. inyectar bloque wiki)
# =============================================================================


def writeback_omdb_wiki_block(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    wiki_block: Mapping[str, object] | None,
) -> bool:
    """
    Escribe en caché OMDb un bloque mínimo de wiki (en __wiki) asociado al record.

    Esto permite que el pipeline “pegue” información de Wikipedia/Wikidata al payload
    de OMDb sin recalcular en cada run.

    Nota:
    - Sanitización y merge del __wiki lo hace omdb_client._merge_dict_shallow().
    - Aquí solo orquestamos el write-back.
    - Se normaliza imdb_id (lower/strip) para maximizar hit en index_imdb.
    """
    try:
        imdb_norm = imdb_id.strip().lower() if isinstance(imdb_id, str) and imdb_id.strip() else None

        if not norm_title and not imdb_norm:
            return False

        patch: dict[str, object] = {"__wiki": dict(wiki_block) if isinstance(wiki_block, Mapping) else None}

        return bool(
            patch_cached_omdb_record(
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
                patch=patch,
            )
        )

    except Exception as exc:
        _dbg(f"writeback_omdb_wiki_block failed: {exc!r}")
        return False


__all__ = [
    "OmdbEnrichment",
    "enrich_with_omdb",
    "apply_omdb_enrichment_to_metadata",
    "writeback_omdb_wiki_block",
]