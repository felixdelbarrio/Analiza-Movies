from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, TypedDict

import requests  # type: ignore[import-not-found]
from requests import Response  # type: ignore[import-not-found]
from requests.adapters import HTTPAdapter  # type: ignore[import-not-found]
from urllib3.util.retry import Retry  # type: ignore[import-not-found]

from backend import logger as _logger
from backend.config import (
    DATA_DIR,
    OMDB_API_KEY,
    OMDB_RATE_LIMIT_MAX_RETRIES,
    OMDB_RATE_LIMIT_WAIT_SECONDS,
    OMDB_RETRY_EMPTY_CACHE,
    SILENT_MODE,
)
from backend.movie_input import normalize_title_for_lookup


# ============================================================
#                       CACHE v2
# ============================================================


class OmdbCacheItem(TypedDict):
    Title: str
    Year: str
    imdbID: str | None
    omdb: dict[str, object]


class OmdbCacheFile(TypedDict):
    schema: int
    items: list[OmdbCacheItem]


_SCHEMA_VERSION: Final[int] = 1
_CACHE_PATH: Final[Path] = DATA_DIR / "omdb_cache.json"

_CACHE: OmdbCacheFile | None = None


# ============================================================
#                  LOGGING CONTROLADO POR SILENT_MODE
# ============================================================


def _log(msg: object) -> None:
    try:
        _logger.info(str(msg))
    except Exception:
        if not SILENT_MODE:
            print(msg)


def _log_always(msg: object) -> None:
    try:
        _logger.warning(str(msg), always=True)
    except Exception:
        print(msg)


# HTTP session con reintentos
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
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
    return session


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
    if not votes or votes == "N/A":
        return None
    if isinstance(votes, (int, float)):
        return int(votes)

    s = str(votes).strip().replace(",", "")
    return _safe_int(s)


def parse_rt_score_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
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


def extract_ratings_from_omdb(
    data: Mapping[str, object] | None,
) -> tuple[float | None, int | None, int | None]:
    if not data:
        return None, None, None

    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)

    return imdb_rating, imdb_votes, rt_score


def is_omdb_data_empty_for_ratings(data: Mapping[str, object] | None) -> bool:
    if not data:
        return True

    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)

    return imdb_rating is None and imdb_votes is None and rt_score is None


def _extract_imdb_id_from_omdb_record(data: Mapping[str, object] | None) -> str | None:
    if not isinstance(data, Mapping):
        return None
    if data.get("Response") != "True":
        return None
    return _safe_imdb_id(data.get("imdbID"))


def _norm_year_str(year: int | None) -> str:
    return str(year) if year is not None else ""


def _is_movie_not_found(data: Mapping[str, object]) -> bool:
    return data.get("Response") == "False" and data.get("Error") == "Movie not found!"


# ============================================================
#           WIKI MINIMAL (SANITIZE PARA NO ENSUCIAR CACHE)
# ============================================================

_ALLOWED_WIKI_KEYS: Final[set[str]] = {"imdb_id", "wikidata_id", "wikipedia_title", "wiki_lookup"}
_ALLOWED_WIKI_LOOKUP_KEYS: Final[set[str]] = {"via", "imdb_id", "title", "year"}


def _sanitize_wiki_lookup(v: object) -> dict[str, object] | None:
    if not isinstance(v, Mapping):
        return None
    out: dict[str, object] = {}

    via = v.get("via")
    if isinstance(via, str) and via.strip():
        out["via"] = via.strip()

    imdb_id = v.get("imdb_id")
    if isinstance(imdb_id, str) and imdb_id.strip():
        out["imdb_id"] = imdb_id.strip()

    title = v.get("title")
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()

    year = v.get("year")
    if isinstance(year, int):
        out["year"] = year
    elif isinstance(year, str) and year.strip().isdigit():
        out["year"] = int(year.strip())

    return out or None


def _sanitize_wiki_block(v: object) -> dict[str, object] | None:
    """
    Deja pasar SOLO:
      - imdb_id (str)
      - wikidata_id (str)
      - wikipedia_title (str)
      - wiki_lookup (dict con {via, imdb_id, title, year})
    """
    if not isinstance(v, Mapping):
        return None

    out: dict[str, object] = {}

    imdb_id = v.get("imdb_id")
    if isinstance(imdb_id, str) and imdb_id.strip():
        out["imdb_id"] = imdb_id.strip()

    wikidata_id = v.get("wikidata_id")
    if isinstance(wikidata_id, str) and wikidata_id.strip():
        out["wikidata_id"] = wikidata_id.strip()

    wikipedia_title = v.get("wikipedia_title")
    if isinstance(wikipedia_title, str) and wikipedia_title.strip():
        out["wikipedia_title"] = wikipedia_title.strip()

    wiki_lookup = _sanitize_wiki_lookup(v.get("wiki_lookup"))
    if wiki_lookup is not None:
        out["wiki_lookup"] = wiki_lookup

    return out or None


# ============================================================
#                  PROVENANCE / TRAZABILIDAD
# ============================================================


def _build_default_provenance(
    *,
    imdb_norm: str | None,
    norm_title: str,
    year: int | None,
) -> dict[str, object]:
    if imdb_norm:
        lookup_key = f"imdb_id:{imdb_norm}"
    else:
        if year is not None:
            lookup_key = f"title_year:{norm_title}|{year}"
        else:
            lookup_key = f"title:{norm_title}"

    return {
        "lookup_key": lookup_key,
        "had_imdb_hint": bool(imdb_norm),
    }


def _merge_provenance(
    existing: Mapping[str, object] | None,
    incoming: Mapping[str, object] | None,
) -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(existing, Mapping):
        out.update(dict(existing))
    if isinstance(incoming, Mapping):
        out.update(dict(incoming))
    return out


def _attach_provenance(
    omdb_data: dict[str, object],
    prov: Mapping[str, object] | None,
) -> dict[str, object]:
    existing = omdb_data.get("__prov")
    merged = _merge_provenance(existing if isinstance(existing, Mapping) else None, prov)
    if merged:
        omdb_data["__prov"] = merged
    return omdb_data


def _merge_dict_shallow(dst: dict[str, object], patch: Mapping[str, object]) -> dict[str, object]:
    """
    Merge “shallow” con comportamiento especial para __prov y __wiki:
      - __prov: mergea dicts
      - __wiki: sanitiza y reemplaza por bloque minimal (o lo elimina si queda vacío)
      - resto: patch pisa claves directas
    """
    for k, v in patch.items():
        if k == "__prov" and isinstance(v, Mapping):
            existing = dst.get("__prov")
            merged = _merge_provenance(existing if isinstance(existing, Mapping) else None, v)
            dst["__prov"] = merged
            continue

        if k == "__wiki":
            sanitized = _sanitize_wiki_block(v)
            if sanitized is None:
                # si el patch trae __wiki pero queda vacío / inválido, mejor no guardarlo
                if "__wiki" in dst:
                    dst.pop("__wiki", None)
            else:
                dst["__wiki"] = sanitized
            continue

        dst[k] = v  # type: ignore[assignment]

    return dst


# ============================================================
#                  LOAD/SAVE CACHE (ATÓMICO)
# ============================================================


def _empty_cache() -> OmdbCacheFile:
    return {"schema": _SCHEMA_VERSION, "items": []}


def _load_cache() -> OmdbCacheFile:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not _CACHE_PATH.exists():
        _CACHE = _empty_cache()
        return _CACHE

    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _CACHE = _empty_cache()
        return _CACHE

    if not isinstance(raw, Mapping):
        _CACHE = _empty_cache()
        return _CACHE

    if raw.get("schema") != _SCHEMA_VERSION:
        _CACHE = _empty_cache()
        return _CACHE

    items_obj = raw.get("items")
    if not isinstance(items_obj, list):
        _CACHE = _empty_cache()
        return _CACHE

    items: list[OmdbCacheItem] = []
    for it in items_obj:
        if not isinstance(it, Mapping):
            continue
        title = it.get("Title")
        year = it.get("Year")
        imdb_id = it.get("imdbID")
        omdb = it.get("omdb")

        if not isinstance(title, str):
            continue
        if not isinstance(year, str):
            continue
        if imdb_id is not None and not isinstance(imdb_id, str):
            continue
        if not isinstance(omdb, Mapping):
            continue

        items.append(
            {
                "Title": title,
                "Year": year,
                "imdbID": imdb_id,
                "omdb": dict(omdb),
            }
        )

    _CACHE = {"schema": _SCHEMA_VERSION, "items": items}
    return _CACHE


def _save_cache(cache: OmdbCacheFile) -> None:
    global _CACHE
    dirpath = _CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(dirpath),
    ) as tf:
        json.dump(cache, tf, ensure_ascii=False, indent=2)
        temp_name = tf.name

    os.replace(temp_name, str(_CACHE_PATH))
    _CACHE = cache


def _find_existing(
    items: list[OmdbCacheItem],
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
) -> OmdbCacheItem | None:
    if imdb_id:
        for it in items:
            if it.get("imdbID") == imdb_id:
                return it
        return None

    for it in items:
        if (
            it.get("imdbID") is None
            and it.get("Title") == norm_title
            and it.get("Year") == norm_year
        ):
            return it

    return None


def iter_cached_omdb_records() -> Iterable[dict[str, object]]:
    cache = _load_cache()
    for it in cache["items"]:
        yield dict(it["omdb"])


# ============================================================
#                 PATCH / WRITE-BACK AL CACHE v2
# ============================================================


def patch_cached_omdb_record(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    patch: Mapping[str, object],
) -> bool:
    """
    Actualiza EN DISCO el registro cacheado (omdb_cache.json) aplicando un patch
    a it["omdb"].

    Matching:
      - Si imdb_id existe: it["imdbID"] == imdb_id
      - Si imdb_id es None: imdbID=None y (Title,Year)==(norm_title,norm_year)

    Merge:
      - "__prov": mergea dicts
      - "__wiki": SANITIZA y guarda solo el bloque minimal permitido
      - resto: pisa claves directas
    """
    imdb_norm = _safe_imdb_id(imdb_id) if imdb_id else None
    cache = _load_cache()
    items = cache.get("items", [])

    target: OmdbCacheItem | None = None
    if imdb_norm:
        for it in items:
            if it.get("imdbID") == imdb_norm:
                target = it
                break
    else:
        for it in items:
            if (
                it.get("imdbID") is None
                and it.get("Title") == norm_title
                and it.get("Year") == norm_year
            ):
                target = it
                break

    if target is None:
        return False

    omdb_obj = target.get("omdb")
    if not isinstance(omdb_obj, dict):
        omdb_obj = dict(omdb_obj) if isinstance(omdb_obj, Mapping) else {}
        target["omdb"] = omdb_obj

    _merge_dict_shallow(omdb_obj, patch)
    _save_cache(cache)
    return True


# ============================================================
#                      FLAGS GLOBAL
# ============================================================

OMDB_DISABLED: bool = False
OMDB_DISABLED_NOTICE_SHOWN: bool = False
OMDB_RATE_LIMIT_NOTICE_SHOWN: bool = False


# ============================================================
#                      PETICIONES OMDb
# ============================================================


def omdb_request(params: Mapping[str, object]) -> dict[str, object] | None:
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


def _get_cached_item(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id_hint: str | None,
) -> OmdbCacheItem | None:
    cache = _load_cache()
    return _find_existing(cache["items"], norm_title, norm_year, imdb_id_hint)


def _cache_store_item(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    omdb_data: dict[str, object],
) -> OmdbCacheItem:
    cache = _load_cache()

    if imdb_id is not None:
        kept: list[OmdbCacheItem] = []
        for it in cache["items"]:
            if it.get("imdbID") == imdb_id:
                continue
            kept.append(it)
        cache["items"] = kept

    if imdb_id is None:
        kept2: list[OmdbCacheItem] = []
        for it in cache["items"]:
            if (
                it.get("imdbID") is None
                and it.get("Title") == norm_title
                and it.get("Year") == norm_year
            ):
                continue
            kept2.append(it)
        cache["items"] = kept2

    item: OmdbCacheItem = {
        "Title": norm_title,
        "Year": norm_year,
        "imdbID": imdb_id,
        "omdb": dict(omdb_data),
    }
    cache["items"].append(item)
    _save_cache(cache)
    return item


def _search_candidates_imdb_id(
    *,
    title_for_search: str,
    year: int | None,
) -> str | None:
    if OMDB_API_KEY is None:
        return None

    if not title_for_search:
        return None

    base_url = "https://www.omdbapi.com/"
    params_s: dict[str, str] = {
        "apikey": OMDB_API_KEY,
        "s": title_for_search,
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

    ptit = normalize_title_for_lookup(title_for_search)

    def score_candidate(cand: Mapping[str, object]) -> float:
        score = 0.0

        ct_raw = cand.get("Title")
        ct = normalize_title_for_lookup(ct_raw) if isinstance(ct_raw, str) else ""
        if ptit and ct:
            if ptit == ct:
                score += 2.0
            elif ct in ptit or ptit in ct:
                score += 1.0

        cand_year: int | None = None
        cy = cand.get("Year")
        if isinstance(cy, str) and cy != "N/A":
            try:
                cand_year = int(cy[:4])
            except Exception:
                cand_year = None

        if year is not None and cand_year is not None:
            if year == cand_year:
                score += 2.0
            elif abs(year - cand_year) <= 1:
                score += 1.0

        return score

    best_imdb: str | None = None
    best_score = float("-inf")

    for item in results_obj:
        if not isinstance(item, Mapping):
            continue
        s = score_candidate(item)
        if s > best_score:
            imdb_raw = _safe_imdb_id(item.get("imdbID"))
            if imdb_raw:
                best_score = s
                best_imdb = imdb_raw

    return best_imdb


def omdb_query_with_cache(
    *,
    title: str | None,
    year: int | None,
    imdb_id: str | None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    global OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN, OMDB_RATE_LIMIT_NOTICE_SHOWN

    imdb_norm = _safe_imdb_id(imdb_id) if imdb_id else None
    year_str = _norm_year_str(year)

    norm_title = normalize_title_for_lookup(title or "")
    if not norm_title and imdb_norm is None:
        return None

    prov_base = _build_default_provenance(imdb_norm=imdb_norm, norm_title=norm_title, year=year)
    prov_in = _merge_provenance(prov_base, provenance)

    cached = _get_cached_item(norm_title=norm_title, norm_year=year_str, imdb_id_hint=imdb_norm)
    if cached is not None and not OMDB_RETRY_EMPTY_CACHE:
        out = dict(cached["omdb"])
        _attach_provenance(out, prov_in)
        return out

    if cached is not None and OMDB_RETRY_EMPTY_CACHE:
        old = dict(cached["omdb"])
        if not is_omdb_data_empty_for_ratings(old):
            _attach_provenance(old, prov_in)
            return old
        _log(f"INFO: reintentando OMDb para {norm_title} ({year_str or '?'}) (cache sin ratings).")

    if OMDB_DISABLED:
        if cached is None:
            return None
        out2 = dict(cached["omdb"])
        _attach_provenance(out2, prov_in)
        return out2

    def _request_with_rate_limit(params: Mapping[str, object]) -> dict[str, object] | None:
        global OMDB_RATE_LIMIT_NOTICE_SHOWN

        retries_local = 0
        while retries_local <= OMDB_RATE_LIMIT_MAX_RETRIES:
            data = omdb_request(params)
            if data is None:
                return None

            if data.get("Error") == "Request limit reached!":
                if not OMDB_RATE_LIMIT_NOTICE_SHOWN:
                    _log_always(
                        "AVISO: límite de llamadas gratuitas de OMDb alcanzado. "
                        f"Esperando {OMDB_RATE_LIMIT_WAIT_SECONDS} segundos antes de continuar..."
                    )
                    OMDB_RATE_LIMIT_NOTICE_SHOWN = True
                    time.sleep(OMDB_RATE_LIMIT_WAIT_SECONDS)
                retries_local += 1
                continue

            return data

        return None

    had_failure = False

    if imdb_norm is not None:
        params_main: dict[str, object] = {"i": imdb_norm, "type": "movie", "plot": "short"}
        data_main = _request_with_rate_limit(params_main)
        if data_main is None:
            had_failure = True
        else:
            imdb_from_resp = _extract_imdb_id_from_omdb_record(data_main)
            imdb_final = imdb_from_resp or imdb_norm

            if not norm_title:
                t_raw = data_main.get("Title")
                if isinstance(t_raw, str):
                    norm_title = normalize_title_for_lookup(t_raw) or norm_title

            if not year_str:
                y_resp = extract_year_from_omdb(data_main)
                if y_resp is not None:
                    year_str = str(y_resp)

            prov_final = dict(prov_in)
            prov_final["resolved_via"] = "i"
            _attach_provenance(data_main, prov_final)

            _cache_store_item(
                norm_title=norm_title,
                norm_year=year_str,
                imdb_id=imdb_final,
                omdb_data=dict(data_main),
            )
            return dict(data_main)

        if had_failure:
            OMDB_DISABLED = True
            if not OMDB_DISABLED_NOTICE_SHOWN:
                _log_always(
                    "ERROR: OMDb desactivado para esta ejecución tras fallos consecutivos. "
                    "A partir de ahora se usará únicamente la caché local."
                )
                OMDB_DISABLED_NOTICE_SHOWN = True

        cached2 = _get_cached_item(norm_title=norm_title, norm_year=year_str, imdb_id_hint=imdb_norm)
        if cached2 is None:
            return None
        out3 = dict(cached2["omdb"])
        _attach_provenance(out3, prov_in)
        return out3

    params_t: dict[str, object] = {"t": norm_title, "type": "movie", "plot": "short"}
    if year is not None:
        params_t["y"] = str(year)

    data_t = _request_with_rate_limit(params_t)
    if data_t is None:
        had_failure = True
    else:
        used_no_year = False
        if year is not None and _is_movie_not_found(data_t):
            params_no_year = dict(params_t)
            params_no_year.pop("y", None)
            data_no_year = _request_with_rate_limit(params_no_year)
            if data_no_year is not None:
                data_t = data_no_year
                used_no_year = True

        if _is_movie_not_found(data_t):
            imdb_best = _search_candidates_imdb_id(title_for_search=norm_title, year=year)
            if imdb_best:
                data_full = _request_with_rate_limit(
                    {"i": imdb_best, "type": "movie", "plot": "short"}
                )
                if isinstance(data_full, dict):
                    imdb_from_full = _extract_imdb_id_from_omdb_record(data_full)
                    imdb_final2 = imdb_from_full or imdb_best

                    y_resp2 = extract_year_from_omdb(data_full)
                    year_str2 = year_str or (str(y_resp2) if y_resp2 is not None else "")

                    prov_final2 = dict(prov_in)
                    prov_final2["resolved_via"] = "s_i"
                    _attach_provenance(data_full, prov_final2)

                    _cache_store_item(
                        norm_title=norm_title,
                        norm_year=year_str2,
                        imdb_id=imdb_final2,
                        omdb_data=dict(data_full),
                    )
                    return dict(data_full)

        imdb_from_resp2 = _extract_imdb_id_from_omdb_record(data_t)
        imdb_final3 = imdb_from_resp2

        y_resp3 = extract_year_from_omdb(data_t)
        year_str3 = year_str or (str(y_resp3) if y_resp3 is not None else "")

        prov_final3 = dict(prov_in)
        if year is not None and not used_no_year:
            prov_final3["resolved_via"] = "t_y"
        else:
            prov_final3["resolved_via"] = "t"
        _attach_provenance(data_t, prov_final3)

        _cache_store_item(
            norm_title=norm_title,
            norm_year=year_str3,
            imdb_id=imdb_final3,
            omdb_data=dict(data_t),
        )
        return dict(data_t)

    if had_failure:
        OMDB_DISABLED = True
        if not OMDB_DISABLED_NOTICE_SHOWN:
            _log_always(
                "ERROR: OMDb desactivado para esta ejecución tras fallos consecutivos. "
                "A partir de ahora se usará únicamente la caché local."
            )
            OMDB_DISABLED_NOTICE_SHOWN = True

    cached3 = _get_cached_item(norm_title=norm_title, norm_year=year_str, imdb_id_hint=None)
    if cached3 is None:
        return None
    out4 = dict(cached3["omdb"])
    _attach_provenance(out4, prov_in)
    return out4


# ============================================================
#                      FUNCIONES PÚBLICAS
# ============================================================


def search_omdb_by_imdb_id(imdb_id: str) -> dict[str, object] | None:
    imdb_norm = _safe_imdb_id(imdb_id)
    if not imdb_norm:
        return None
    return omdb_query_with_cache(title=None, year=None, imdb_id=imdb_norm)


def search_omdb_by_title_and_year(
    title: str,
    year: int | None,
) -> dict[str, object] | None:
    if not title.strip():
        return None
    return omdb_query_with_cache(title=title, year=year, imdb_id=None)


def search_omdb_with_candidates(
    plex_title: str,
    plex_year: int | None,
) -> dict[str, object] | None:
    title_raw = plex_title.strip()
    if not title_raw:
        return None

    data = search_omdb_by_title_and_year(title_raw, plex_year)
    if data and data.get("Response") == "True":
        return data

    title_for_search = normalize_title_for_lookup(title_raw)
    imdb_best = _search_candidates_imdb_id(title_for_search=title_for_search, year=plex_year)
    if not imdb_best:
        return None

    return search_omdb_by_imdb_id(imdb_best)