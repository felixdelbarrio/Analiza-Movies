from __future__ import annotations

"""
backend/analiza_omdb.py

Orquestador de enriquecimiento OMDb (sin UI).

Responsabilidades
-----------------
- Invocar el cliente OMDb/cache (backend.omdb_client) de forma robusta (best-effort).
- Normalizar/extraer campos útiles (ratings/votes/rt/metacritic) desde el payload OMDb.
- Exponer una estructura “digerida” para el resto del pipeline.
- (Opcional) Persistir "parches" a records cacheados vía patch_cached_omdb_record
  para evitar recalcular/reenriquecer en futuras ejecuciones.

Compat extract_ratings_from_omdb
-------------------------------
- Acepta:
    - versión antigua: (imdb_rating, imdb_votes, rt_score)
    - versión nueva:   (imdb_rating, imdb_votes, rt_score, metacritic_score)

API pública
-----------
- enrich_with_omdb(title, year, imdb_id, provenance=None) -> OmdbEnrichment | None
- apply_omdb_enrichment_to_metadata(metadata, enrichment, overwrite=False) -> dict (in-place shallow)
- writeback_omdb_wiki_block(norm_title, norm_year, imdb_id, wiki_block) -> bool
"""

from collections.abc import Mapping
from dataclasses import dataclass

from backend import logger
from backend.title_utils import normalize_title_for_lookup
from backend.omdb_client import extract_ratings_from_omdb, omdb_query_with_cache, patch_cached_omdb_record


# =============================================================================
# Logging (centralizado)
# =============================================================================


def _dbg(msg: object) -> None:
    """Diagnóstico contextual (solo si DEBUG_MODE=True). Best-effort."""
    fn = getattr(logger, "debug_ctx", None)
    if callable(fn):
        try:
            fn("OMDB", msg)
        except Exception:
            pass


# =============================================================================
# Data model (lo que el resto del pipeline realmente necesita)
# =============================================================================


@dataclass(frozen=True)
class OmdbEnrichment:
    """
    Resultado “digerido” para el pipeline.

    - payload: dict OMDb completo (para consumidores avanzados)
    - imdb_rating, imdb_votes, rt_score, metacritic_score: valores parseados (None si no disponibles)
    - has_ratings: True si existe al menos uno de los 4 (rating/votes/rt/metacritic)
    - imdb_id: imdbID final (si OMDb lo devuelve)
    - norm_title, norm_year: canon de cache/pipeline (normalizados)
    """

    payload: dict[str, object]
    imdb_rating: float | None
    imdb_votes: int | None
    rt_score: int | None
    metacritic_score: int | None
    has_ratings: bool
    imdb_id: str | None
    norm_title: str
    norm_year: str


# =============================================================================
# Helpers
# =============================================================================


def _extract_ratings_compat(data: Mapping[str, object]) -> tuple[float | None, int | None, int | None, int | None]:
    """
    Compatibilidad con extract_ratings_from_omdb:
      - viejo: (imdb_rating, imdb_votes, rt_score)
      - nuevo: (imdb_rating, imdb_votes, rt_score, metacritic_score)
    """
    try:
        out_obj: object = extract_ratings_from_omdb(data)
    except Exception:
        return None, None, None, None

    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None
    mc_score: int | None = None

    if isinstance(out_obj, tuple):
        if len(out_obj) >= 1 and (isinstance(out_obj[0], (int, float)) or out_obj[0] is None):
            imdb_rating = float(out_obj[0]) if isinstance(out_obj[0], (int, float)) else None
        if len(out_obj) >= 2 and (isinstance(out_obj[1], int) or out_obj[1] is None):
            imdb_votes = out_obj[1]
        if len(out_obj) >= 3 and (isinstance(out_obj[2], int) or out_obj[2] is None):
            rt_score = out_obj[2]
        if len(out_obj) >= 4 and (isinstance(out_obj[3], int) or out_obj[3] is None):
            mc_score = out_obj[3]

    return imdb_rating, imdb_votes, rt_score, mc_score


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
    - Response=False => None (sin enrichment), pero en DEBUG loguea el error.
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

        if data.get("Response") != "True":
            err = data.get("Error")
            if err is not None:
                _dbg(f"OMDb Response=False: {err!r}")
            return None

        imdb_rating, imdb_votes, rt_score, metacritic_score = _extract_ratings_compat(data)
        has_ratings = not (
            imdb_rating is None and imdb_votes is None and rt_score is None and metacritic_score is None
        )

        imdb_id_final = data.get("imdbID")
        if not isinstance(imdb_id_final, str) or not imdb_id_final.strip():
            imdb_id_final = None
        else:
            imdb_id_final = imdb_id_final.strip().lower()

        t = data.get("Title")
        if isinstance(t, str) and t.strip():
            norm_title = normalize_title_for_lookup(t) or norm_title

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
            metacritic_score=metacritic_score,
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
    - omdb_imdb_rating, omdb_imdb_votes, omdb_rt_score, omdb_metacritic_score
    - omdb_has_ratings (bool)
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
        put("omdb_metacritic_score", enrichment.metacritic_score)

        put("omdb_has_ratings", bool(enrichment.has_ratings))
        put("imdbID", enrichment.imdb_id)

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
    """Escribe en caché OMDb un bloque mínimo de wiki (en __wiki) asociado al record."""
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