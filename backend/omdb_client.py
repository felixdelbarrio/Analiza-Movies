from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend import logger as _logger
from backend.config import (
    OMDB_API_KEY,
    OMDB_RATE_LIMIT_WAIT_SECONDS,
    OMDB_RATE_LIMIT_MAX_RETRIES,
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
#                      CACHE OMDb LOCAL
# ============================================================

CACHE_FILE: Final[str] = "omdb_cache.json"
CACHE_PATH: Final[Path] = Path(CACHE_FILE)


def load_cache() -> dict[str, object]:
    """
    Carga la caché de OMDb desde disco y mantiene compatibilidad
    con el formato antiguo.
    """
    if not CACHE_PATH.exists():
        return {}

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            raw_cache = json.load(f)
    except Exception as exc:
        _log(f"WARNING: error cargando {CACHE_PATH}: {exc}")
        # Intentar renombrar el archivo corrupto para preservar datos
        try:
            broken = CACHE_PATH.with_suffix(".broken.json")
            CACHE_PATH.replace(broken)
            _log(f"INFO: archivo de cache corrupto renombrado a {broken}")
        except Exception:
            pass
        return {}

    if not isinstance(raw_cache, dict):
        return {}

    normalized: dict[str, object] = dict(raw_cache)

    for key, value in list(raw_cache.items()):
        if not isinstance(key, str):
            continue

        stripped = key.strip()

        # Si la clave parece un JSON antiguo → convertir
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                params = json.loads(stripped)
            except Exception:
                continue

            if not isinstance(params, dict):
                continue

            imdb_id_obj = params.get("i")
            title_obj = params.get("t")
            year_obj = params.get("y")

            imdb_id = imdb_id_obj if isinstance(imdb_id_obj, str) else None
            title = title_obj if isinstance(title_obj, str) else None
            year = year_obj if isinstance(year_obj, str) else None

            canon_key: str | None
            if imdb_id:
                canon_key = imdb_id
            elif title:
                title_low = title.lower()
                if year:
                    canon_key = f"title:{year}:{title_low}"
                else:
                    canon_key = f"title::{title_low}"
            else:
                canon_key = None

            if canon_key and canon_key not in normalized:
                normalized[canon_key] = value

    return normalized


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
            json.dump(cache, tf, indent=2, ensure_ascii=False)
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
    ni puntuación de Rotten Tomatoes. Se usa para saber si una
    entrada de caché está "vacía" a efectos de scoring.
    """
    if not data:
        return True

    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)

    return imdb_rating is None and imdb_votes is None and rt_score is None


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
    req_params: dict[str, str] = {}
    for key, val in params.items():
        req_params[str(key)] = str(val)
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
    - Cache OMDb.
    - OMDB_RETRY_EMPTY_CACHE.
    - Reintentos por rate limit.
    - Desactivación global de OMDb.

    Además:
    - Cuando OMDb responde "Request limit reached!", mostramos SIEMPRE
      un aviso (una sola vez) y esperamos OMDB_RATE_LIMIT_WAIT_SECONDS
      antes de seguir.
    """
    global OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN, OMDB_RATE_LIMIT_NOTICE_SHOWN

    # Si OMDb está desactivado → solo cache
    if OMDB_DISABLED:
        cached = omdb_cache.get(cache_key)
        return cached if isinstance(cached, dict) else None

    # Cache hit (normal, sin reintento)
    if cache_key in omdb_cache and not OMDB_RETRY_EMPTY_CACHE:
        cached = omdb_cache[cache_key]
        return cached if isinstance(cached, dict) else None

    # Cache hit pero se quiere volver a intentar porque estaba "vacía" de ratings
    if cache_key in omdb_cache and OMDB_RETRY_EMPTY_CACHE:
        old = omdb_cache[cache_key]
        if isinstance(old, dict) and not is_omdb_data_empty_for_ratings(old):
            return old
        _log(f"INFO: reintentando OMDb para {cache_key} (cache sin ratings).")

    # Reintentos reales a la API
    retries = 0
    had_failure = False

    while retries <= OMDB_RATE_LIMIT_MAX_RETRIES:
        data = omdb_request(params)

        if data is None:
            had_failure = True
        else:
            error_msg = data.get("Error")

            # ----------------------------
            # Caso: límite gratuito OMDb
            # ----------------------------
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

            # ----------------------------
            # Caso normal con datos (OK o 'Movie not found')
            # ----------------------------
            if data.get("Response") == "True":
                omdb_cache[cache_key] = data
                save_cache(omdb_cache)
                return data

            # Error no fatal (por ejemplo Movie not found) → almacenar y devolver
            omdb_cache[cache_key] = data
            save_cache(omdb_cache)
            return data

        retries += 1

    # ------------------------------------------------------------------
    # Si agotamos todos los reintentos → desactivamos OMDb para la sesión
    # ------------------------------------------------------------------
    if had_failure:
        OMDB_DISABLED = True
        if not OMDB_DISABLED_NOTICE_SHOWN:
            _log_always(
                "ERROR: OMDb desactivado para esta ejecución tras fallos consecutivos. "
                "A partir de ahora se usará únicamente la caché local."
            )
            OMDB_DISABLED_NOTICE_SHOWN = True

    cached = omdb_cache.get(cache_key)
    return cached if isinstance(cached, dict) else None


# ============================================================
#                      FUNCIONES PÚBLICAS
# ============================================================


def search_omdb_by_imdb_id(imdb_id: str) -> dict[str, object] | None:
    if not imdb_id:
        return None

    cache_key = imdb_id
    params: dict[str, object] = {"i": imdb_id, "type": "movie", "plot": "short"}
    return omdb_query_with_cache(cache_key, params)


def search_omdb_by_title_and_year(
    title: str,
    year: int | None,
) -> dict[str, object] | None:
    if not title:
        return None

    cache_key = f"title:{year}:{title.lower()}" if year else f"title::{title.lower()}"
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
        cache_key_no_year = f"title::{title.lower()}"
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

    # 1) búsqueda directa título+year
    data = search_omdb_by_title_and_year(title, plex_year)
    if data and data.get("Response") == "True":
        return data

    # 2) búsqueda por 's' de OMDb (búsqueda libre)
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

        # coincidencia parcial por palabras
        common = sum(1 for w in ptit.split() if w and w in ctit)
        score += common * 0.1

        return score

    # elegir el mejor candidato
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

    imdb_id_obj = best_dict.get("imdbID")
    imdb_id = imdb_id_obj if isinstance(imdb_id_obj, str) else None
    if not imdb_id:
        return None

    return search_omdb_by_imdb_id(imdb_id)