from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeAlias

from backend import logger as _logger
from backend.analyze_input_core import AnalysisRow, analyze_input_movie
from backend.metadata_fix import generate_metadata_suggestions_row
from backend.movie_input import MovieInput
from backend.omdb_client import omdb_query_with_cache
from backend.wiki_client import get_wiki_client

# ✅ Write-back a cache OMDb (si existe en tu omdb_client.py)
try:
    from backend.omdb_client import patch_cached_omdb_record  # type: ignore
except Exception:  # pragma: no cover
    patch_cached_omdb_record = None  # type: ignore[assignment]

# Para normalizar el title como hace omdb_client internamente
try:
    from backend.movie_input import normalize_title_for_lookup
except Exception:  # pragma: no cover
    normalize_title_for_lookup = None  # type: ignore[assignment]


_OmdbCacheKey: TypeAlias = tuple[str, int | None, str | None]

_OMDB_LOCAL_CACHE: dict[_OmdbCacheKey, dict[str, object]] = {}
_WIKI_LOCAL_CACHE: dict[_OmdbCacheKey, dict[str, object]] = {}


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.upper() == "N/A":
            return None
        s = s.replace(",", "")
        return int(s)
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return None
    except Exception:
        return None


def _extract_wiki_meta(omdb_record: Mapping[str, object] | None) -> dict[str, object]:
    if not omdb_record:
        return {}
    wiki_raw = omdb_record.get("__wiki")
    if isinstance(wiki_raw, Mapping):
        return dict(wiki_raw)
    return {}


def _build_lookup_key(title_for_fetch: str, year_for_fetch: int | None, imdb_hint: str | None) -> str:
    t = title_for_fetch.strip()
    if imdb_hint:
        return f"imdb_id:{imdb_hint}"
    if year_for_fetch is not None:
        return f"title_year:{t}|{year_for_fetch}"
    return f"title:{t}"


def _build_wiki_lookup_info(
    *,
    title_for_fetch: str,
    year_for_fetch: int | None,
    imdb_used: str | None,
) -> dict[str, object]:
    """
    Describe con qué parámetros se consultó a Wiki.
    """
    title_clean = title_for_fetch.strip()
    if imdb_used:
        return {"via": "imdb_id", "imdb_id": imdb_used}
    if year_for_fetch is not None:
        return {"via": "title_year", "title": title_clean, "year": year_for_fetch}
    return {"via": "title", "title": title_clean}


def _build_minimal_wiki_block(
    *,
    imdb_id: str | None,
    wikidata_id: str | None,
    wikipedia_title: str | None,
    wiki_lookup: Mapping[str, object],
) -> dict[str, object]:
    """
    ✅ Bloque mínimo que sí queremos persistir en omdb_cache.json.
    """
    out: dict[str, object] = {
        "wiki_lookup": dict(wiki_lookup),
    }
    if imdb_id is not None:
        out["imdb_id"] = imdb_id
    if wikidata_id is not None:
        out["wikidata_id"] = wikidata_id
    if wikipedia_title is not None:
        out["wikipedia_title"] = wikipedia_title
    return out


def _persist_minimal_wiki_into_omdb_cache(
    *,
    title_for_fetch: str,
    year_for_fetch: int | None,
    imdb_id_for_cache: str | None,
    imdb_hint: str | None,
    minimal_wiki: Mapping[str, object],
    lookup_key: str,
) -> None:
    """
    Persiste __wiki MINIMAL y wiki_wikidata_id en __prov dentro del omdb_cache.json.

    Requiere patch_cached_omdb_record(); si no existe, no hace nada.
    """
    if patch_cached_omdb_record is None:
        return

    imdb_id_final = imdb_id_for_cache or imdb_hint

    if normalize_title_for_lookup is not None:
        norm_title = normalize_title_for_lookup(title_for_fetch or "")
    else:
        norm_title = (title_for_fetch or "").strip().lower()

    norm_year = str(year_for_fetch) if year_for_fetch is not None else ""

    wikidata_id = None
    if isinstance(minimal_wiki, Mapping):
        raw = minimal_wiki.get("wikidata_id")
        wikidata_id = raw if isinstance(raw, str) and raw.strip() else None

    prov_patch: dict[str, object] = {
        "lookup_key": lookup_key,
        "had_imdb_hint": bool(imdb_hint),
    }
    if wikidata_id:
        prov_patch["wiki_wikidata_id"] = wikidata_id

    patch_cached_omdb_record(
        norm_title=norm_title,
        norm_year=norm_year,
        imdb_id=imdb_id_final,
        patch={
            "__wiki": dict(minimal_wiki),
            "__prov": prov_patch,  # en omdb_client mergea __prov, no lo pisa a lo bruto
        },
    )


def analyze_movie(
    movie_input: MovieInput,
    *,
    source_movie: object | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[str]]:
    logs: list[str] = []

    # ------------------------------------------------------------------
    # 0) Precedencia de título/año para reporting (Plex manda si está)
    # ------------------------------------------------------------------
    display_title_raw = movie_input.extra.get("display_title")
    display_year_raw = movie_input.extra.get("display_year")

    display_title = (
        display_title_raw
        if isinstance(display_title_raw, str) and display_title_raw.strip()
        else movie_input.title
    )

    display_year: int | None
    if isinstance(display_year_raw, int):
        display_year = display_year_raw
    else:
        display_year = movie_input.year

    # ------------------------------------------------------------------
    # 1) Fetch OMDb + Wiki (con caché global)
    # ------------------------------------------------------------------
    omdb_data: Mapping[str, object] | None = None
    wiki_meta: dict[str, object] = {}

    def _norm_imdb_hint() -> str | None:
        hint = movie_input.imdb_id_hint
        if isinstance(hint, str):
            h = hint.strip()
            return h or None
        return None

    def fetch_omdb(title_for_fetch: str, year_for_fetch: int | None) -> Mapping[str, object]:
        nonlocal omdb_data, wiki_meta

        imdb_hint = _norm_imdb_hint()
        key: _OmdbCacheKey = (title_for_fetch, year_for_fetch, imdb_hint)

        # ✅ 1) Cache local (módulo)
        cached = _OMDB_LOCAL_CACHE.get(key)
        if cached is not None:
            omdb_data = cached
            wiki_meta = _WIKI_LOCAL_CACHE.get(key, {})
            return cached

        lookup_key = _build_lookup_key(title_for_fetch, year_for_fetch, imdb_hint)

        # ✅ 2) OMDb (cache v2 del propio cliente) + provenance embebida
        omdb_record = omdb_query_with_cache(
            title=title_for_fetch,
            year=year_for_fetch,
            imdb_id=imdb_hint,
            provenance={
                "lookup_key": lookup_key,
                "had_imdb_hint": bool(imdb_hint),
            },
        )

        if omdb_record is None:
            empty: dict[str, object] = {}
            _OMDB_LOCAL_CACHE[key] = empty
            _WIKI_LOCAL_CACHE[key] = {}
            omdb_data = empty
            wiki_meta = {}
            return empty

        omdb_dict = dict(omdb_record)

        # ✅ 3) Si OMDb YA trae __wiki (pero ojo: ahora es minimal), no llamamos a Wiki
        if "__wiki" in omdb_dict and isinstance(omdb_dict.get("__wiki"), Mapping):
            wiki_dict = _extract_wiki_meta(omdb_dict)
            _OMDB_LOCAL_CACHE[key] = omdb_dict
            _WIKI_LOCAL_CACHE[key] = wiki_dict
            omdb_data = omdb_dict
            wiki_meta = wiki_dict
            return omdb_dict

        # ✅ 4) Si ya teníamos wiki_cache local, lo aplicamos y persistimos (minimal)
        cached_wiki = _WIKI_LOCAL_CACHE.get(key)
        if cached_wiki is not None and cached_wiki:
            omdb_dict["__wiki"] = dict(cached_wiki)

            imdb_id_from_omdb: str | None = None
            imdb_raw = omdb_dict.get("imdbID")
            if isinstance(imdb_raw, str) and imdb_raw.strip():
                imdb_id_from_omdb = imdb_raw.strip()

            _persist_minimal_wiki_into_omdb_cache(
                title_for_fetch=title_for_fetch,
                year_for_fetch=year_for_fetch,
                imdb_id_for_cache=imdb_id_from_omdb,
                imdb_hint=imdb_hint,
                minimal_wiki=cached_wiki,
                lookup_key=lookup_key,
            )

            _OMDB_LOCAL_CACHE[key] = omdb_dict
            omdb_data = omdb_dict
            wiki_meta = dict(cached_wiki)
            return omdb_dict

        # --- Wiki: intentamos con título+año (y con imdb si lo hay) ---
        imdb_id_from_omdb: str | None = None
        imdb_raw = omdb_dict.get("imdbID")
        if isinstance(imdb_raw, str) and imdb_raw.strip():
            imdb_id_from_omdb = imdb_raw.strip()

        imdb_used_for_wiki = imdb_id_from_omdb or imdb_hint

        wiki_item = get_wiki_client().get_wiki(
            title=title_for_fetch,
            year=year_for_fetch,
            imdb_id=imdb_used_for_wiki,
        )

        # ✅ Construimos siempre un bloque MINIMAL si Wiki responde
        if wiki_item is not None:
            wiki_block = wiki_item.get("wiki")
            wikidata_block = wiki_item.get("wikidata")

            # imdb_id que nos devuelve wiki_client (si lo ofrece)
            imdb_cached = wiki_item.get("imdbID")
            imdb_from_wiki = imdb_cached if isinstance(imdb_cached, str) and imdb_cached.strip() else None

            # wikidata_id: lo buscamos primero en wikidata_block (típico)
            wikidata_id: str | None = None
            if isinstance(wikidata_block, Mapping):
                # soporta varias formas por compatibilidad
                for k in ("wikibase_item", "wikidata_id", "id"):
                    v = wikidata_block.get(k)
                    if isinstance(v, str) and v.strip():
                        wikidata_id = v.strip()
                        break

            # wikipedia_title: lo buscamos en wiki_block si viene
            wikipedia_title: str | None = None
            if isinstance(wiki_block, Mapping):
                v = wiki_block.get("wikipedia_title")
                if isinstance(v, str) and v.strip():
                    wikipedia_title = v.strip()

            wiki_lookup = _build_wiki_lookup_info(
                title_for_fetch=title_for_fetch,
                year_for_fetch=year_for_fetch,
                imdb_used=imdb_used_for_wiki,
            )

            minimal_wiki = _build_minimal_wiki_block(
                imdb_id=imdb_from_wiki or imdb_used_for_wiki,
                wikidata_id=wikidata_id,
                wikipedia_title=wikipedia_title,
                wiki_lookup=wiki_lookup,
            )

            omdb_dict["__wiki"] = minimal_wiki

            # ✅ Persistimos inmediatamente en cache OMDb (minimal + prov.wiki_wikidata_id)
            _persist_minimal_wiki_into_omdb_cache(
                title_for_fetch=title_for_fetch,
                year_for_fetch=year_for_fetch,
                imdb_id_for_cache=imdb_id_from_omdb,
                imdb_hint=imdb_hint,
                minimal_wiki=minimal_wiki,
                lookup_key=lookup_key,
            )

        wiki_dict = _extract_wiki_meta(omdb_dict)

        _OMDB_LOCAL_CACHE[key] = omdb_dict
        _WIKI_LOCAL_CACHE[key] = wiki_dict

        omdb_data = omdb_dict
        wiki_meta = wiki_dict
        return omdb_dict

    # ------------------------------------------------------------------
    # 2) Metacritic final (OMDb -> Wiki)
    # ------------------------------------------------------------------
    record_for_meta = fetch_omdb(movie_input.title, movie_input.year)
    metacritic_score = _safe_int(record_for_meta.get("Metascore"))
    if metacritic_score is None and wiki_meta:
        metacritic_score = _safe_int(wiki_meta.get("metacritic_score"))

    # ------------------------------------------------------------------
    # 3) Plex rating (si aplica)
    # ------------------------------------------------------------------
    plex_rating: float | None = None
    if movie_input.source == "plex" and source_movie is not None:
        plex_user_rating = getattr(source_movie, "userRating", None)
        plex_rating_raw = getattr(source_movie, "rating", None)
        plex_rating = _safe_float(plex_user_rating) or _safe_float(plex_rating_raw)

    # ------------------------------------------------------------------
    # 4) Core genérico: ratings + decisión + misidentified_hint (único)
    # ------------------------------------------------------------------
    try:
        base_row: AnalysisRow = analyze_input_movie(
            movie_input,
            fetch_omdb,
            plex_title=display_title,
            plex_year=display_year,
            plex_rating=plex_rating,
            metacritic_score=metacritic_score,
        )
    except Exception as exc:  # pragma: no cover
        msg = (
            f"[ERROR] {movie_input.library} / {display_title} ({display_year}): "
            f"fallo en core de análisis: {exc}"
        )
        _logger.error(msg)
        logs.append(msg)
        return None, None, logs

    if not base_row:
        logs.append(
            f"[WARN] {movie_input.library} / {display_title} ({display_year}): "
            "core de análisis devolvió fila vacía."
        )
        return None, None, logs

    misidentified_hint = base_row.get("misidentified_hint")
    if isinstance(misidentified_hint, str) and misidentified_hint:
        logs.append(
            f"[MISIDENTIFIED] {movie_input.library} / {display_title} ({display_year}): "
            f"{misidentified_hint}"
        )

    # ------------------------------------------------------------------
    # 5) Sugerencias de metadata (solo Plex) -> usamos MovieInput
    # ------------------------------------------------------------------
    omdb_dict: dict[str, object] = dict(omdb_data) if omdb_data else {}
    meta_sugg: dict[str, object] | None = None
    if movie_input.source == "plex" and source_movie is not None:
        try:
            meta_candidate = generate_metadata_suggestions_row(
                movie_input,
                omdb_dict or None,
            )
            if isinstance(meta_candidate, dict):
                meta_sugg = meta_candidate
                logs.append(
                    "[METADATA_SUGG] "
                    f"{movie_input.library} / {display_title} ({display_year}): "
                    f"{meta_sugg.get('suggestions_json', '')}"
                )
        except Exception as exc:  # pragma: no cover
            _logger.warning(
                f"generate_metadata_suggestions_row falló para {display_title!r}: {exc}"
            )

    # ------------------------------------------------------------------
    # 6) Enriquecimiento estándar para reporting
    # ------------------------------------------------------------------
    poster_url: str | None = None
    trailer_url: str | None = None
    imdb_id: str | None = None
    omdb_json_str: str | None = None

    if omdb_dict:
        poster_raw = omdb_dict.get("Poster")
        trailer_raw = omdb_dict.get("Website")
        imdb_id_raw = omdb_dict.get("imdbID")

        poster_url = poster_raw if isinstance(poster_raw, str) else None
        trailer_url = trailer_raw if isinstance(trailer_raw, str) else None
        if isinstance(imdb_id_raw, str):
            imdb_id = imdb_id_raw

    imdb_hint = _norm_imdb_hint()
    if imdb_id is None and imdb_hint is not None:
        imdb_id = imdb_hint

    if omdb_dict:
        try:
            omdb_json_str = json.dumps(omdb_dict, ensure_ascii=False)
        except Exception:
            omdb_json_str = str(omdb_dict)

    # ------------------------------------------------------------------
    # 7) Construcción fila final (Plex prevalece en campos nativos)
    # ------------------------------------------------------------------
    row: dict[str, object] = dict(base_row)

    row["source"] = movie_input.source
    row["library"] = movie_input.library
    row["title"] = display_title
    row["year"] = display_year

    file_size_bytes = row.get("file_size_bytes")
    if isinstance(file_size_bytes, int):
        row["file_size"] = file_size_bytes
    else:
        row["file_size"] = movie_input.file_size_bytes

    row["file"] = movie_input.file_path or row.get("file", "")

    source_url_obj = movie_input.extra.get("source_url")
    row["source_url"] = source_url_obj if isinstance(source_url_obj, str) else ""

    row["rating_key"] = movie_input.rating_key
    row["guid"] = movie_input.plex_guid
    row["thumb"] = movie_input.thumb_url

    row["imdb_id"] = imdb_id
    row["poster_url"] = poster_url
    row["trailer_url"] = trailer_url
    row["omdb_json"] = omdb_json_str

    # ✅ Ahora wiki_meta es minimal: wikidata_id y wikipedia_title salen directo remindando compat
    row["wikidata_id"] = wiki_meta.get("wikadata_id") or wiki_meta.get("wikidata_id")
    row["wikipedia_title"] = wiki_meta.get("wikipedia_title")
    row["source_language"] = wiki_meta.get("source_language")

    return row, meta_sugg, logs