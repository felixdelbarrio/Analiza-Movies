from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.config import OMDB_RETRY_EMPTY_CACHE, SILENT_MODE
from backend.omdb_client import (
    search_omdb_by_imdb_id,
    search_omdb_with_candidates,
    omdb_cache,
    is_omdb_data_empty_for_ratings,
)
from backend import logger as _logger

# --------------------------------------------------------------------
# Tipos auxiliares
# --------------------------------------------------------------------


class WikiMeta(TypedDict, total=False):
    wikidata_id: str | None
    wikipedia_title: str | None
    source_lang: str | None
    imdb_id: str | None


WikiRecord = dict[str, object]
WikiCache = dict[str, WikiRecord]


# --------------------------------------------------------------------
# Fichero de caché maestro (wiki + omdb fusionado)
# --------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
WIKI_CACHE_PATH: Path = BASE_DIR / "wiki_cache.json"

_wiki_cache: WikiCache = {}
_wiki_cache_loaded: bool = False

# Contexto de progreso para prefijar logs (x/total, biblioteca, título)
_CURRENT_PROGRESS: dict[str, object | None] = {
    "idx": None,
    "total": None,
    "library": None,
    "title": None,
}


def set_wiki_progress(idx: int, total: int, library_title: str, movie_title: str) -> None:
    """
    Se llama desde analiza_plex para que los logs de wiki_client sepan
    en qué punto del análisis estamos y puedan prefijar (x/total).
    """
    _CURRENT_PROGRESS["idx"] = idx
    _CURRENT_PROGRESS["total"] = total
    _CURRENT_PROGRESS["library"] = library_title
    _CURRENT_PROGRESS["title"] = movie_title


def _progress_prefix() -> str:
    idx = _CURRENT_PROGRESS.get("idx")
    total = _CURRENT_PROGRESS.get("total")
    library = _CURRENT_PROGRESS.get("library")
    title = _CURRENT_PROGRESS.get("title")

    if (
        not isinstance(idx, int)
        or not isinstance(total, int)
        or not isinstance(library, str)
        or not isinstance(title, str)
    ):
        return ""

    return f"({idx}/{total}) {library} · {title} | "


def _log_wiki(msg: str) -> None:
    """
    Log interno de wiki_client controlado por SILENT_MODE.
    """
    prefix = _progress_prefix()
    text = f"{prefix}{msg}"
    try:
        _logger.info(text)
    except Exception:
        if not SILENT_MODE:
            print(text)


# --------------------------------------------------------------------
# HTTP session with retries
# --------------------------------------------------------------------
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a cached requests.Session configured with retries/backoff."""
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


# --------------------------------------------------------------------
# Carga / guardado de wiki_cache
# --------------------------------------------------------------------
def _load_wiki_cache() -> None:
    global _wiki_cache_loaded, _wiki_cache
    if _wiki_cache_loaded:
        return

    if WIKI_CACHE_PATH.exists():
        try:
            with WIKI_CACHE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # No tipamos más profundo aquí; asumimos dict[str, WikiRecord]
                _wiki_cache = {str(k): v for k, v in data.items()}
            else:
                _wiki_cache = {}
            _log_wiki(f"[WIKI] wiki_cache cargada ({len(_wiki_cache)} entradas)")
        except Exception as exc:
            _logger.warning(f"[WIKI] Error cargando wiki_cache.json: {exc}")
            # Try to preserve the broken file before resetting cache
            try:
                broken = WIKI_CACHE_PATH.with_suffix(".broken.json")
                WIKI_CACHE_PATH.replace(broken)
                _logger.warning(f"[WIKI] Archivo corrupto renombrado a {broken}")
            except Exception:
                # best-effort, ignore
                pass
            _wiki_cache = {}
    else:
        _wiki_cache = {}

    _wiki_cache_loaded = True


def _save_wiki_cache() -> None:
    if not _wiki_cache_loaded:
        return
    try:
        # Write atomically to avoid corrupting cache on crash
        dirpath = WIKI_CACHE_PATH.parent
        dirpath.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(dirpath),
        ) as tf:
            json.dump(_wiki_cache, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name
        os.replace(temp_name, str(WIKI_CACHE_PATH))
    except Exception as exc:
        _logger.error(f"[WIKI] Error guardando wiki_cache.json: {exc}")


def _normalize_title(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


# --------------------------------------------------------------------
# Consultas a Wikidata / Wikipedia
# --------------------------------------------------------------------
WIKIDATA_API: str = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL: str = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_TEMPLATE: str = "https://{lang}.wikipedia.org/w/api.php"


def _wikidata_get_entity(wikidata_id: str) -> dict[str, object] | None:
    try:
        session = _get_session()
        resp = session.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": wikidata_id,
                "format": "json",
                "props": "labels|claims|sitelinks",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities", {})
        if isinstance(entities, dict):
            entity = entities.get(wikidata_id)
            if isinstance(entity, dict):
                return entity
        return None
    except Exception as exc:
        _log_wiki(f"[WIKI] Error obteniendo entidad {wikidata_id} de Wikidata: {exc}")
        return None


def _wikidata_search_by_imdb(imdb_id: str) -> tuple[str, dict[str, object]] | None:
    """
    Busca en Wikidata por propiedad P345 (IMDb ID).
    Devuelve (wikidata_id, entity) o None.
    """
    query = f"""
    SELECT ?item WHERE {{
      ?item wdt:P345 "{imdb_id}" .
    }} LIMIT 1
    """

    try:
        session = _get_session()
        resp = session.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", {}).get("bindings", [])
        if not isinstance(results, list) or not results:
            return None

        first = results[0]
        item = first.get("item", {})
        if not isinstance(item, dict):
            return None

        item_uri = item.get("value")
        if not isinstance(item_uri, str):
            return None

        wikidata_id = item_uri.split("/")[-1]
        entity = _wikidata_get_entity(wikidata_id)
        if not entity:
            return None

        return wikidata_id, entity
    except Exception as exc:
        _log_wiki(f"[WIKI] Error en búsqueda SPARQL por IMDb ID {imdb_id}: {exc}")
        return None


def _wikidata_search_by_title(
    title: str,
    year: int | None,
    language: str = "en",
) -> tuple[str, dict[str, object]] | None:
    """
    Busca una película por título (y opcionalmente año) usando wbsearchentities.
    Filtra por tipo 'film' y comprueba fecha aproximada de publicación.
    """
    try:
        session = _get_session()
        resp = session.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": title,
                "language": language,
                "type": "item",
                "format": "json",
                "limit": 5,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("search", [])
        if not isinstance(search_results, list) or not search_results:
            return None

        for candidate in search_results:
            if not isinstance(candidate, dict):
                continue

            wikidata_id = candidate.get("id")
            if not isinstance(wikidata_id, str):
                continue

            entity = _wikidata_get_entity(wikidata_id)
            if not entity:
                continue

            claims = entity.get("claims", {})
            if not isinstance(claims, dict):
                continue

            # P31 = instance of → film / animated film
            if "P31" in claims:
                ok_type = False
                for inst in claims["P31"]:
                    if not isinstance(inst, dict):
                        continue
                    val = (
                        inst.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                    )
                    if isinstance(val, dict) and val.get("id") in {
                        "Q11424",  # film
                        "Q24869",  # animated film
                    }:
                        ok_type = True
                        break
                if not ok_type:
                    continue

            # P577 = publication date
            if year is not None and "P577" in claims:
                try:
                    first_p577 = claims["P577"][0]
                    if isinstance(first_p577, dict):
                        time_str = (
                            first_p577.get("mainsnak", {})
                            .get("datavalue", {})
                            .get("value", {})
                            .get("time")
                        )
                        if isinstance(time_str, str) and len(time_str) >= 5:
                            ent_year = int(time_str[1:5])
                            # Permitimos ±1 año
                            if abs(ent_year - int(year)) > 1:
                                continue
                except Exception:
                    # Si falla, no descartamos por año
                    pass

            return wikidata_id, entity

        return None
    except Exception as exc:
        _log_wiki(f"[WIKI] Error buscando por título '{title}' en Wikidata: {exc}")
        return None


def _extract_imdb_id_from_entity(entity: dict[str, object]) -> str | None:
    claims = entity.get("claims", {})
    if not isinstance(claims, dict) or "P345" not in claims:
        return None
    try:
        first = claims["P345"][0]
        if not isinstance(first, dict):
            return None
        snak = first["mainsnak"]
        if not isinstance(snak, dict):
            return None
        datavalue = snak["datavalue"]
        if not isinstance(datavalue, dict):
            return None
        value = datavalue["value"]
        if not isinstance(value, str):
            return None
        return value
    except Exception:
        return None


def _extract_wikipedia_title(entity: dict[str, object], language: str) -> str | None:
    sitelinks = entity.get("sitelinks", {})
    if not isinstance(sitelinks, dict):
        return None
    key = f"{language}wiki"
    site = sitelinks.get(key)
    if isinstance(site, dict):
        title = site.get("title")
        return str(title) if title is not None else None
    return None


# --------------------------------------------------------------------
# API pública: get_movie_record
# --------------------------------------------------------------------


def get_movie_record(
    title: str,
    year: int | None = None,
    imdb_id_hint: str | None = None,
    language: str = "en",
) -> WikiRecord | None:
    """
    Devuelve un registro "tipo OMDb" enriquecido con metadata de Wikipedia/Wikidata,
    usando wiki_cache.json como master.

    Reglas clave:

      - Si NO hay imdb_id_hint:
          * SIEMPRE se consulta Wikidata para intentar obtener imdb_id.
          * Con ese imdb_id se completa vía OMDb (cache → API).

      - Si SÍ hay imdb_id_hint:
          * OMDB_RETRY_EMPTY_CACHE = False → no se consulta Wikidata
            (solo OMDb / omdb_cache).
          * OMDB_RETRY_EMPTY_CACHE = True  → sí se consulta Wikidata
            para enriquecer con __wiki.

      - Si existe entrada en wiki_cache:
          * Se devuelve directamente.
          * Si OMDB_RETRY_EMPTY_CACHE = True y no tiene ratings,
            se reintenta OMDb con el imdb_id y se actualiza wiki_cache.
    """
    _load_wiki_cache()

    norm_title = _normalize_title(title)
    if imdb_id_hint:
        base_cache_key = f"imdb:{imdb_id_hint.lower()}"
    else:
        base_cache_key = f"title:{year}:{norm_title}"

    _log_wiki(
        f"[WIKI] get_movie_record(title='{title}', year={year}, imdb_hint={imdb_id_hint}) "
        f"→ base_cache_key='{base_cache_key}'"
    )

    # ----------------------------------------------------------------
    # 0) HIT en wiki_cache
    # ----------------------------------------------------------------
    record = _wiki_cache.get(base_cache_key)
    if record is not None:
        _log_wiki(f"[WIKI] cache HIT para {base_cache_key}")

        if OMDB_RETRY_EMPTY_CACHE and is_omdb_data_empty_for_ratings(record):
            imdb_cached = record.get("imdbID") or (
                record.get("__wiki", {}) if isinstance(record.get("__wiki"), dict) else {}
            )
            if isinstance(imdb_cached, dict):
                imdb_cached_val = imdb_cached.get("imdb_id")
            else:
                imdb_cached_val = imdb_cached

            if isinstance(imdb_cached_val, str) and imdb_cached_val:
                _log_wiki(
                    f"[WIKI] Registro en wiki_cache sin ratings, reintentando OMDb "
                    f"con imdbID={imdb_cached_val}..."
                )
                refreshed = search_omdb_by_imdb_id(imdb_cached_val)
                if refreshed and refreshed.get("Response") == "True":
                    merged: WikiRecord = dict(refreshed)
                    wiki_part = record.get("__wiki")
                    if isinstance(wiki_part, dict):
                        merged["__wiki"] = wiki_part
                    _wiki_cache[base_cache_key] = merged
                    _save_wiki_cache()
                    _log_wiki(
                        f"[WIKI] wiki_cache actualizada con datos OMDb para {base_cache_key}"
                    )
                    return merged

        return record

    _log_wiki(f"[WIKI] cache MISS para {base_cache_key}, resolviendo...")

    # ----------------------------------------------------------------
    # 1) Resolver Wikidata / imdb_id_final según reglas
    # ----------------------------------------------------------------
    wikidata_id: str | None = None
    entity: dict[str, object] | None = None
    imdb_id_from_wiki: str | None = None
    wiki_title: str | None = None
    source_lang = language

    # Caso: tenemos imdb_id_hint y NO queremos tocar Wikidata
    if imdb_id_hint and not OMDB_RETRY_EMPTY_CACHE:
        _log_wiki(
            "[WIKI] imdb_id_hint disponible y OMDB_RETRY_EMPTY_CACHE=False → "
            "saltando Wikidata."
        )
        imdb_id_final = imdb_id_hint
        wiki_meta: WikiMeta = {}
    else:
        # 1a) Intentar por IMDb ID
        if imdb_id_hint:
            result = _wikidata_search_by_imdb(imdb_id_hint)
            if result is not None:
                wikidata_id, entity = result
                imdb_id_from_wiki = _extract_imdb_id_from_entity(entity)
                wiki_title = (
                    _extract_wikipedia_title(entity, language)
                    or _extract_wikipedia_title(entity, "es")
                )
                _log_wiki(
                    f"[WIKI] Encontrado en Wikidata por IMDb ID {imdb_id_hint}: "
                    f"wikidata_id={wikidata_id}, imdb_id_wiki={imdb_id_from_wiki}, "
                    f"wikipedia_title='{wiki_title}'"
                )

        # 1b) Si no hay entidad aún, buscar por título + año
        if entity is None:
            for lang in (language, "es"):
                result = _wikidata_search_by_title(title, year, language=lang)
                if result is not None:
                    wikidata_id, entity = result
                    imdb_id_from_wiki = _extract_imdb_id_from_entity(entity)
                    wiki_title = _extract_wikipedia_title(entity, lang)
                    source_lang = lang
                    _log_wiki(
                        "[WIKI] Encontrada película en Wikidata por título: "
                        f"wikidata_id={wikidata_id}, imdb_id_wiki={imdb_id_from_wiki}, "
                        f"wikipedia_title='{wiki_title}', lang={lang}"
                    )
                    break

        wiki_meta: WikiMeta = {}
        if entity is not None:
            wiki_meta = {
                "wikidata_id": wikidata_id,
                "wikipedia_title": wiki_title,
                "source_lang": source_lang,
                "imdb_id": imdb_id_from_wiki or imdb_id_hint,
            }

        imdb_id_final = imdb_id_from_wiki or imdb_id_hint

    # ----------------------------------------------------------------
    # 2) Resolver datos OMDb
    # ----------------------------------------------------------------
    omdb_data: dict[str, object] | None

    if imdb_id_final:
        omdb_data = search_omdb_by_imdb_id(imdb_id_final)
    else:
        omdb_data = search_omdb_with_candidates(title, year)

    # ----------------------------------------------------------------
    # 3) Construir registro final y guardar en wiki_cache
    # ----------------------------------------------------------------
    if omdb_data:
        record_out: WikiRecord = dict(omdb_data)
        if imdb_id_final and not record_out.get("imdbID"):
            record_out["imdbID"] = imdb_id_final
    else:
        record_out = {
            "Title": title,
            "Year": str(year) if year is not None else None,
            "imdbID": imdb_id_final,
        }

    if wiki_meta:
        record_out["__wiki"] = wiki_meta

    # Clave preferente basada en imdbID, para reutilizar entre variantes de título
    if imdb_id_final:
        wiki_key = f"imdb:{imdb_id_final.lower()}"
    else:
        wiki_key = base_cache_key

    _wiki_cache[wiki_key] = record_out
    if wiki_key != base_cache_key:
        _wiki_cache[base_cache_key] = record_out

    _save_wiki_cache()

    wikidata_id_logged: str | None = None
    wiki_part = record_out.get("__wiki")
    if isinstance(wiki_part, dict):
        wd_val = wiki_part.get("wikidata_id")
        if isinstance(wd_val, str):
            wikidata_id_logged = wd_val

    _log_wiki(
        f"[WIKI] Registro maestro creado para {wiki_key}: "
        f"imdbID={record_out.get('imdbID')}, "
        f"wikidata_id={wikidata_id_logged}"
    )

    return record_out