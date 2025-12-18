from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend import logger as _logger
from backend.config import (
    DATA_DIR,
    OMDB_API_KEY,
    OMDB_RATE_LIMIT_MAX_RETRIES,
    OMDB_RATE_LIMIT_WAIT_SECONDS,
    OMDB_RETRY_EMPTY_CACHE,
    SILENT_MODE,
)

# ============================================================
#                  LOGGING CONTROLADO POR SILENT_MODE
# ============================================================


def _log(msg: object) -> None:
    """Logea vía logger central respetando SILENT_MODE."""
    try:
        _logger.info(str(msg))
    except Exception:
        if not SILENT_MODE:
            print(msg)


def _log_always(msg: object) -> None:
    """Log crítico que siempre se muestra (usa logger.warning with always)."""
    try:
        _logger.warning(str(msg), always=True)
    except Exception:
        print(msg)


# HTTP session con reintentos
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    """Devuelve una sesión HTTP con reintentos configurados."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _SESSION = session
    return _SESSION


# ============================================================
#                 FUNCIONES AUXILIARES VARIAS
# ============================================================


def _safe_int(s: object) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _safe_float(s: object) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_imdb_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    return v.lower()


def normalize_imdb_votes(votes: object) -> int | None:
    """
    Convierte el campo votes de OMDb (por ejemplo "123,456") en int.
    Devuelve None si no se puede parsear.
    """
    if not votes or votes == "N/A":
        return None
    if isinstance(votes, (int, float)):
        return int(votes)

    s = str(votes).strip().replace(",", "")
    return _safe_int(s)


def parse_rt_score_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    """
    Busca el rating de Rotten Tomatoes en Ratings y lo devuelve 0-100.
    """
    ratings_obj = omdb_data.get("Ratings") or []
    if not isinstance(ratings_obj, list):
        return None

    for r in ratings_obj:
        if not isinstance(r, Mapping):
            continue
        source = r.get("Source")
        if source != "Rotten Tomatoes":
            continue
        val = r.get("Value")
        if not isinstance(val, str):
            continue
        if not val.endswith("%"):
            continue
        try:
            return int(val[:-1])
        except ValueError:
            return None
    return None


def parse_imdb_rating_from_omdb(omdb_data: Mapping[str, object]) -> float | None:
    raw = omdb_data.get("imdbRating")
    if not raw or raw == "N/A":
        return None
    return _safe_float(raw)


def extract_year_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    raw = omdb_data.get("Year")
    if not raw or raw == "N/A":
        return None
    text = str(raw).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


# ============================================================
#                      CACHE OMDb LOCAL (imdb único)
# ============================================================

CACHE_FILE: Final[str] = "omdb_cache.json"
CACHE_PATH: Final[Path] = DATA_DIR / CACHE_FILE


class _AliasRecord(TypedDict):
    __alias: str


def _is_alias_record(obj: object) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("__alias"), str)


def _resolve_cache_entry(
    cache: dict[str, object],
    key: str,
    *,
    max_depth: int = 10,
) -> dict[str, object] | None:
    """
    Resuelve una clave en cache soportando alias:
      - Canon: "tt123..." -> dict OMDb
      - Alias: "title:..." -> {"__alias": "tt123..."}
    """
    current_key = key
    seen: set[str] = set()

    for _ in range(max_depth):
        if current_key in seen:
            return None
        seen.add(current_key)

        raw = cache.get(current_key)
        if raw is None:
            return None
        if _is_alias_record(raw):
            target = raw.get("__alias")
            if not isinstance(target, str) or not target.strip():
                return None
            current_key = target.strip().lower()
            continue

        return dict(raw) if isinstance(raw, dict) else None

    return None


def _extract_imdb_id_from_omdb_record(data: Mapping[str, object] | None) -> str | None:
    if not isinstance(data, Mapping):
        return None
    if data.get("Response") != "True":
        return None
    return _safe_imdb_id(data.get("imdbID"))


def load_cache() -> dict[str, object]:
    """
    Carga la caché OMDb desde disco.

    Modelo:
      - Canon: imdbID en minúsculas ("tt...") -> dict OMDb completo
      - Alias: "title:..." -> {"__alias": "tt..."}
      - Para Response=False se guarda bajo la clave de consulta (title:*), sin alias.
    """
    if not CACHE_PATH.exists():
        return {}

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            raw_cache = json.load(f)
    except Exception as exc:
        _log(f"WARNING: error cargando {CACHE_PATH}: {exc}")
        try:
            broken = CACHE_PATH.with_suffix(".broken.json")
            CACHE_PATH.replace(broken)
            _log(f"INFO: archivo de cache corrupto renombrado a {broken}")
        except Exception:
            pass
        return {}

    if not isinstance(raw_cache, dict):
        return {}

    # Normalización mínima (claves string)
    normalized: dict[str, object] = {}
    for k, v in raw_cache.items():
        if not isinstance(k, str):
            continue
        normalized[k.strip()] = v

    # Compatibilidad parcial con claves antiguas JSON-serializadas (si existieran aún)
    for key, value in list(normalized.items()):
        stripped = key.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue

        try:
            params = json.loads(stripped)
        except Exception:
            continue
        if not isinstance(params, dict):
            continue

        imdb_id = params.get("i") if isinstance(params.get("i"), str) else None
        title = params.get("t") if isinstance(params.get("t"), str) else None
        year = params.get("y") if isinstance(params.get("y"), str) else None

        canon_key: str | None
        if imdb_id:
            canon_key = imdb_id.strip().lower()
        elif title:
            title_low = title.lower()
            canon_key = f"title:{year}:{title_low}" if year else f"title::{title_low}"
        else:
            canon_key = None

        if canon_key and canon_key not in normalized:
            normalized[canon_key] = value

    # Migración/normalización a imdb único + alias
    upgraded: dict[str, object] = {}
    changed = False

    for key, value in list(normalized.items()):
        if not isinstance(key, str):
            continue

        if _is_alias_record(value):
            target = value.get("__alias")
            if isinstance(target, str) and target.strip():
                upgraded[key] = _AliasRecord(__alias=target.strip().lower())
            else:
                # alias inválido -> ignorar
                changed = True
            continue

        if not isinstance(value, dict):
            upgraded[key] = value
            continue

        imdb_id = _extract_imdb_id_from_omdb_record(value)
        if imdb_id is None:
            upgraded[key] = dict(value)
            continue

        # Canon por imdbID
        canon_existing = upgraded.get(imdb_id)
        if canon_existing is None or _is_alias_record(canon_existing):
            upgraded[imdb_id] = dict(value)
            changed = True

        # Alias si la clave no es el canon
        key_norm = key.strip().lower()
        if key_norm != imdb_id:
            upgraded[key] = _AliasRecord(__alias=imdb_id)
            changed = True
        else:
            upgraded[key_norm] = dict(value)

    if changed:
        save_cache(upgraded)
        _log("INFO: omdb_cache migrada/normalizada a formato alias (imdb único).")

    return upgraded


def save_cache(cache: Mapping[str, object]) -> None:
    """Escribe la cache de forma atómica en CACHE_PATH."""
    dirpath = CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(dirpath),
        ) as tf:
            json.dump(dict(cache), tf, indent=2, ensure_ascii=False)
            temp_name = tf.name
        os.replace(temp_name, str(CACHE_PATH))
    except Exception as exc:
        _log(f"ERROR guardando cache OMDb en {CACHE_PATH}: {exc}")


omdb_cache: dict[str, object] = load_cache()

# Flags globales
OMDB_DISABLED: bool = False
OMDB_DISABLED_NOTICE_SHOWN: bool = False
OMDB_RATE_LIMIT_NOTICE_SHOWN: bool = False  # para el aviso de límite gratuito


# ============================================================
#                  EXTRACCIÓN DE RATINGS
# ============================================================


def extract_ratings_from_omdb(
    data: Mapping[str, object] | None,
) -> tuple[float | None, int | None, int | None]:
    """
    Extrae imdb_rating, imdb_votes y rt_score de un dict OMDb.
    """
    if not data:
        return None, None, None

    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)

    return imdb_rating, imdb_votes, rt_score


def is_omdb_data_empty_for_ratings(data: Mapping[str, object] | None) -> bool:
    """
    Devuelve True si el dict OMDb no tiene rating IMDb, ni votos,
    ni puntuación de Rotten Tomatoes.
    """
    if not data:
        return True

    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)

    return imdb_rating is None and imdb_votes is None and rt_score is None


def _persist_alias(cache_key: str, imdb_id: str) -> None:
    """
    Guarda alias title:* -> tt... (imdb único). No escribe si coincide.
    """
    ck = cache_key.strip()
    if not ck:
        return
    if ck.lower() == imdb_id:
        return
    omdb_cache[ck] = _AliasRecord(__alias=imdb_id)


# ============================================================
#                      PETICIONES OMDb
# ============================================================


def omdb_request(params: Mapping[str, object]) -> dict[str, object] | None:
    """
    Petición directa sin cache.
    """
    if OMDB_API_KEY is None:
        _log("ERROR: OMDB_API_KEY no configurada.")
        return None

    if OMDB_DISABLED:
        return None

    base_url = "https://www.omdbapi.com/"
    req_params: dict[str, str] = {str(k): str(v) for k, v in params.items()}
    req_params["apikey"] = OMDB_API_KEY

    try:
        session = _get_session()
        resp: Response = session.get(base_url, params=req_params, timeout=10)
    except Exception as exc:
        _log(f"WARNING: error al conectar con OMDb: {exc}")
        return None

    if resp.status_code != 200:
        _log(f"WARNING: OMDb devolvió status {resp.status_code}")
        return None

    try:
        data_obj = resp.json()
    except Exception as exc:
        _log(f"WARNING: OMDb no devolvió JSON válido: {exc}")
        return None

    if isinstance(data_obj, dict):
        return data_obj
    _log("WARNING: OMDb devolvió JSON no dict.")
    return None


def omdb_query_with_cache(
    cache_key: str,
    params: Mapping[str, object],
) -> dict[str, object] | None:
    """
    Gestiona:
    - Cache OMDb (con alias).
    - OMDB_RETRY_EMPTY_CACHE.
    - Reintentos por rate limit.
    - Desactivación global de OMDb.
    """
    global OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN, OMDB_RATE_LIMIT_NOTICE_SHOWN

    cache_key_norm = cache_key.strip().lower()

    # Si OMDb está desactivado → solo cache (resolviendo alias)
    if OMDB_DISABLED:
        return _resolve_cache_entry(omdb_cache, cache_key_norm)

    # Cache hit (normal, sin reintento)
    if cache_key_norm in omdb_cache and not OMDB_RETRY_EMPTY_CACHE:
        return _resolve_cache_entry(omdb_cache, cache_key_norm)

    # Cache hit pero se quiere reintentar porque estaba "vacía" de ratings
    if cache_key_norm in omdb_cache and OMDB_RETRY_EMPTY_CACHE:
        old = _resolve_cache_entry(omdb_cache, cache_key_norm)
        if old is not None and not is_omdb_data_empty_for_ratings(old):
            return old
        _log(f"INFO: reintentando OMDb para {cache_key_norm} (cache sin ratings).")

    retries = 0
    had_failure = False

    while retries <= OMDB_RATE_LIMIT_MAX_RETRIES:
        data = omdb_request(params)

        if data is None:
            had_failure = True
        else:
            error_msg = data.get("Error")

            if error_msg == "Request limit reached!":
                had_failure = True

                if not OMDB_RATE_LIMIT_NOTICE_SHOWN:
                    _log_always(
                        "AVISO: límite de llamadas gratuitas de OMDb alcanzado. "
                        f"Esperando {OMDB_RATE_LIMIT_WAIT_SECONDS} segundos antes de continuar..."
                    )
                    OMDB_RATE_LIMIT_NOTICE_SHOWN = True
                    time.sleep(OMDB_RATE_LIMIT_WAIT_SECONDS)

                retries += 1
                continue

            if data.get("Response") == "True":
                imdb_id = _extract_imdb_id_from_omdb_record(data)
                if imdb_id:
                    omdb_cache[imdb_id] = dict(data)
                    _persist_alias(cache_key_norm, imdb_id)
                else:
                    # Caso raro: Response=True pero sin imdbID
                    omdb_cache[cache_key_norm] = dict(data)

                save_cache(omdb_cache)
                return dict(data)

            # Response=False (Movie not found, etc.)
            omdb_cache[cache_key_norm] = dict(data)
            save_cache(omdb_cache)
            return dict(data)

        retries += 1

    if had_failure:
        OMDB_DISABLED = True
        if not OMDB_DISABLED_NOTICE_SHOWN:
            _log_always(
                "ERROR: OMDb desactivado para esta ejecución tras fallos consecutivos. "
                "A partir de ahora se usará únicamente la caché local."
            )
            OMDB_DISABLED_NOTICE_SHOWN = True

    return _resolve_cache_entry(omdb_cache, cache_key_norm)


# ============================================================
#                      FUNCIONES PÚBLICAS
# ============================================================


def search_omdb_by_imdb_id(imdb_id: str) -> dict[str, object] | None:
    imdb_norm = _safe_imdb_id(imdb_id)
    if not imdb_norm:
        return None

    cache_key = imdb_norm
    params: dict[str, object] = {"i": imdb_norm, "type": "movie", "plot": "short"}
    return omdb_query_with_cache(cache_key, params)


def search_omdb_by_title_and_year(
    title: str,
    year: int | None,
) -> dict[str, object] | None:
    if not title:
        return None

    title_low = title.lower()
    cache_key = f"title:{year}:{title_low}" if year else f"title::{title_low}"
    params: dict[str, object] = {"t": title, "type": "movie", "plot": "short"}
    if year is not None:
        params["y"] = str(year)

    data = omdb_query_with_cache(cache_key, params)

    # Reintento sin año si OMDb dice "Movie not found!"
    if (
        data
        and data.get("Response") == "False"
        and data.get("Error") == "Movie not found!"
    ):
        cache_key_no_year = f"title::{title_low}"
        params_no_year = dict(params)
        params_no_year.pop("y", None)
        data = omdb_query_with_cache(cache_key_no_year, params_no_year)

    return data


def search_omdb_with_candidates(
    plex_title: str,
    plex_year: int | None,
) -> dict[str, object] | None:
    """
    Último recurso cuando:
      - No se obtuvo IMDb ID
      - No se encuentra título exacto

    Estrategia:
      1) Buscar por título+year.
      2) Si falla, usar 's=' de OMDb y elegir el mejor candidato por heurística.
    """
    title = plex_title.strip()
    if not title:
        return None

    data = search_omdb_by_title_and_year(title, plex_year)
    if data and data.get("Response") == "True":
        return data

    if OMDB_API_KEY is None:
        _log("ERROR: OMDB_API_KEY no configurada para búsqueda de candidatos.")
        return None

    base_url = "https://www.omdbapi.com/"
    params_s: dict[str, str] = {
        "apikey": OMDB_API_KEY,
        "s": title,
        "type": "movie",
    }

    try:
        session = _get_session()
        resp: Response = session.get(base_url, params=params_s, timeout=10)
        data_s = resp.json() if resp.status_code == 200 else None
    except Exception:
        data_s = None

    if not isinstance(data_s, dict) or data_s.get("Response") != "True":
        return None

    results_obj = data_s.get("Search") or []
    if not isinstance(results_obj, list):
        return None

    def score_candidate(cand: Mapping[str, object]) -> float:
        score = 0.0

        title_obj = cand.get("Title")
        ctit = title_obj.lower() if isinstance(title_obj, str) else ""
        ptit = title.lower()

        if ptit == ctit:
            score += 2.0
        elif ctit in ptit or ptit in ctit:
            score += 1.0

        cand_year: int | None = None
        cy = cand.get("Year")
        if isinstance(cy, str) and cy != "N/A":
            try:
                cand_year = int(cy[:4])
            except Exception:
                cand_year = None

        if plex_year is not None and cand_year is not None:
            if plex_year == cand_year:
                score += 2.0
            elif abs(plex_year - cand_year) <= 1:
                score += 1.0

        common = sum(1 for w in ptit.split() if w and w in ctit)
        score += common * 0.1

        return score

    best_dict: dict[str, object] | None = None
    best_score = float("-inf")

    for item in results_obj:
        if not isinstance(item, Mapping):
            continue
        cand_score = score_candidate(item)
        if cand_score > best_score:
            best_score = cand_score
            best_dict = dict(item)

    if not best_dict:
        return None

    imdb_id = _safe_imdb_id(best_dict.get("imdbID"))
    if not imdb_id:
        return None

    return search_omdb_by_imdb_id(imdb_id)