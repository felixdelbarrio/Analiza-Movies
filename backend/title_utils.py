"""
backend/title_utils.py

Utilidades neutrales y reutilizables para:
- Normalización de títulos (lookup vs compare)
- Heurísticas de idioma / escritura (CJK)
- Limpieza conservadora de "ruido" típico de releases
- Extractores puros desde texto/path: año e imdb tt*

Objetivo:
- Evitar duplicación entre movie_input.py, decision_logic.py, metadata_fix.py, etc.
- No hacer logging (módulo core/utility).
- Evitar dependencias circulares: este módulo NO importa MovieInput.

Nota:
- Este módulo usa knobs definidos en backend.config_plex para modular comportamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import unicodedata
from typing import Final

from backend.config_plex import (
    MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS,
    MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK,
    MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE,
    MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP,
    MOVIE_INPUT_LOOKUP_STRIP_ACCENTS,
)

# ============================================================================
# Regex/constantes comunes
# ============================================================================

# Año (captura 1900-2099) con separadores comunes
_YEAR_IN_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s\(\[\{._\-])((?:19|20)\d{2})(?:$|[\s\)\]\}._\-])"
)

# Año al final tipo: "Title (1999)" o "Title - 1999" o "Title.1999"
_YEAR_TRAILING_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^
    (?P<title>.*?)
    (?:\s*[\(\[\{]\s*(?P<year1>(?:19|20)\d{2})\s*[\)\]\}]\s*
      |\s*[-._]\s*(?P<year2>(?:19|20)\d{2})\s*
    )
    $
    """,
    re.VERBOSE,
)

# IMDb ID tt1234567 en cualquier parte
_IMDB_TT_RE: Final[re.Pattern[str]] = re.compile(r"\b(tt\d{7,9})\b", re.IGNORECASE)

_MULTI_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s{2,}")

# Para lookup: conservamos algunos acentos antes de strip opcional
_LOOKUP_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(
    r"[^0-9A-Za-záéíóúÁÉÍÓÚñÑüÜ]+", re.UNICODE
)

# Para compare: agresivo y estable (latin básico + dígitos)
_COMPARE_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9\s]")

_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Separadores típicos de releases
_SEP_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("_", " "),
    (".", " "),
    ("\u00a0", " "),  # NBSP
    ("–", "-"),
    ("—", "-"),
)

# ============================================================================
# Noise tokens: releases/filenames (codec, rip, edition, audio, subs, tags)
# ============================================================================

_NOISE_SINGLE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^(?:
        480p|576p|720p|1080p|1440p|2160p|4320p|
        4k|8k|uhd|hdr|hdr10|dv|

        x264|x265|h\.?264|h\.?265|hevc|avc|

        bluray|blu-?ray|bdrip|brrip|dvdrip|dvd|web-?dl|webrip|hdrip|
        cam|ts|tc|scr|dvdscr|r5|

        proper|repack|remux|limited|unrated|extended|cut|

        multi|dual|vose|vos|
        castellano|espa[nñ]ol|spanish|latino|
        eng|english|
        ita|italian|italiano|
        fra|fre|french|fran[cç]ais|
        jpn|jap|japanese|nihongo|
        kor|korean|
        chi|zho|chinese|

        ac3|dts|aac|flac|truehd|atmos|

        subs?|subbed|

        yify|rarbg
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NOISE_PHRASE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        dolby\s*vision|
        director'?s\s*cut|
        dual(?:\s*audio)?|
        hdr\s*10
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ============================================================================
# Secuelas / romanos (variantes de búsqueda)
# ============================================================================

_SEQUEL_PART_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:part|parte|pt)\s+(?:i|ii|iii|iv|v|vi|vii|viii|ix|\d+)\s*$",
    re.IGNORECASE,
)
_SEQUEL_ROMAN_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:i|ii|iii|iv|v|vi|vii|viii|ix)\s*$", re.IGNORECASE
)
_SEQUEL_TRAILING_PART_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:part|parte|pt)\s*$", re.IGNORECASE
)
# ============================================================================
# Heurística de idioma / escritura (reutilizable)
# ============================================================================

_ES_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[ñÑáéíóúÁÉÍÓÚüÜ]")

_EN_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(the|a|an|and|or|of|to|in|on|for|with|without|from|by|part|chapter|episode)\b",
    re.IGNORECASE,
)

# Señal ligera de NO-EN (ES/FR/IT comunes) para detectar títulos localizados
_NON_EN_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"el|la|los|las|un|una|unos|unas|de|del|y|o|en|con|sin|por|para|al|lo|le|les|"
    r"du|des|et|ou|une|"
    r"di|da|e|il|gli"
    r")\b",
    re.IGNORECASE,
)

# Unicode blocks (pistas fuertes por escritura; no equivalen 1:1 a idioma)
_KANA_RE: Final[re.Pattern[str]] = re.compile(
    r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F]"
)
_HANGUL_RE: Final[re.Pattern[str]] = re.compile(
    r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]"
)
_HAN_RE: Final[re.Pattern[str]] = re.compile(r"[\u4E00-\u9FFF]")
_CJK_ANY_RE: Final[re.Pattern[str]] = re.compile(
    r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F\u4E00-\u9FFF\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]"
)

# ============================================================================
# Helpers puros
# ============================================================================


def strip_accents(text: str) -> str:
    """Elimina diacríticos (NFKD)."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def cleanup_separators(text: str) -> str:
    """Normaliza separadores típicos de filenames y guiones unicode."""
    out = text or ""
    for a, b in _SEP_REPLACEMENTS:
        out = out.replace(a, b)
    return out


def title_has_cjk_script(text: str) -> bool:
    """True si el texto contiene escritura CJK/Hangul."""
    return bool(text and _CJK_ANY_RE.search(text))


def is_probably_english_title(title: str) -> bool:
    """
    Heurística ligera: True si el título parece inglés.
    (Evita CJK y evita “acentos españoles” como señal negativa básica.)

    Nota:
    - Está sesgada a detectar inglés por "function words".
    - Títulos ingleses muy cortos ("Jaws", "Rocky") pueden devolver False.
    """
    t = (title or "").strip()
    if not t:
        return False
    if _ES_CHAR_RE.search(t):
        return False
    if _CJK_ANY_RE.search(t):
        return False

    words = cleanup_separators(t.lower())
    threshold = int(MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS)
    if threshold <= 0:
        return bool(_EN_FUNCTION_WORD_RE.search(words))
    return len(_EN_FUNCTION_WORD_RE.findall(words)) >= threshold


def is_probably_non_english_title(title: str) -> bool:
    """
    Heurística ligera: True si el título parece NO inglés (ES/FR/IT),
    útil para detectar localización frente a OMDb (frecuentemente EN).
    """
    t = (title or "").strip()
    if not t:
        return False
    if _CJK_ANY_RE.search(t):
        return False

    # Diacríticos “españoles” (y similares) => fuerte NO-EN
    if _ES_CHAR_RE.search(t):
        return True

    words = cleanup_separators(t.lower())
    return bool(_NON_EN_FUNCTION_WORD_RE.search(words))


def should_skip_title_similarity_due_to_language(
    plex_title: str, omdb_title: str
) -> bool:
    """
    Guard para evitar comparar títulos localizados (ES/FR/IT, etc.) contra OMDb (a menudo EN),
    ya que la similitud será baja y generará falsos “diverge”.
    """
    pt = (plex_title or "").strip()
    ot = (omdb_title or "").strip()
    if not pt or not ot:
        return True

    if MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK and (
        title_has_cjk_script(pt) or title_has_cjk_script(ot)
    ):
        return True

    omdb_is_en = is_probably_english_title(ot)
    plex_is_en = is_probably_english_title(pt)

    plex_non_en = is_probably_non_english_title(pt)
    omdb_non_en = is_probably_non_english_title(ot)

    if (omdb_is_en and (plex_non_en or not plex_is_en)) or (
        plex_is_en and (omdb_non_en or not omdb_is_en)
    ):
        return True

    return bool(omdb_is_en and not plex_is_en)


def generate_sequel_title_variants(title: str) -> list[str]:
    """
    Genera variantes simples sin sufijos de secuela (I/II/III o Parte II).
    """
    raw = (title or "").strip()
    if not raw:
        return []

    out: list[str] = []
    seen: set[str] = set()
    lower_raw = raw.lower()

    t_part = _SEQUEL_PART_SUFFIX_RE.sub("", raw).strip()
    if t_part and t_part.lower() != lower_raw:
        key = t_part.lower()
        if key not in seen:
            seen.add(key)
            out.append(t_part)

    t_roman = _SEQUEL_ROMAN_SUFFIX_RE.sub("", raw).strip()
    if t_roman and t_roman.lower() != lower_raw:
        if _SEQUEL_TRAILING_PART_RE.search(t_roman):
            return out
        key = t_roman.lower()
        if key not in seen:
            seen.add(key)
            out.append(t_roman)

    return out


def looks_like_noise_group(text: str) -> bool:
    """
    Decide si un bloque entre [](){} parece “ruido” técnico.
    Conservador: ante dudas, preferimos NO eliminar.
    """
    t = (text or "").strip()
    if not t:
        return True

    if _NOISE_PHRASE_RE.search(t):
        return True

    toks = cleanup_separators(t).split()
    for tok in toks:
        if _NOISE_SINGLE_TOKEN_RE.match(tok):
            return True

    if t.isdigit():
        return True
    if _YEAR_IN_TEXT_RE.search(t):
        return True

    return False


def _strip_bracket_groups(text: str, pat: re.Pattern[str]) -> str:
    """
    Sustituye grupos bracketed detectados como ruido por espacios.
    (Fix: antes se detenía en el primer grupo "no-ruido" y dejaba basura posterior.)
    """
    if not text:
        return text

    def repl(m: re.Match[str]) -> str:
        g = m.group(0)
        if len(g) < 2:
            return g
        inner = g[1:-1]
        return " " if looks_like_noise_group(inner) else g

    return pat.sub(repl, text)


def remove_bracketed_noise(text: str) -> str:
    """
    Elimina grupos entre [] () {} cuando parecen ruido.
    Controlado por MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE.
    """
    if not MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE:
        return text

    out = text
    out = _strip_bracket_groups(out, re.compile(r"\[[^\]]+\]"))
    out = _strip_bracket_groups(out, re.compile(r"\([^\)]+\)"))
    out = _strip_bracket_groups(out, re.compile(r"\{[^\}]+\}"))
    return out


def remove_trailing_dash_group(text: str) -> str:
    """
    Si el título es "Movie - 1080p - x265" y lo de la derecha parece ruido, recortamos.
    Controlado por MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP.
    """
    if not MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP:
        return text

    parts = [p.strip() for p in (text or "").split(" - ")]
    if len(parts) <= 1:
        return text

    left = parts[0].strip()
    if not left:
        return text

    right = " ".join(parts[1:]).strip()
    if looks_like_noise_group(right):
        return left

    return text


def remove_noise_tokens(text: str) -> str:
    """Elimina tokens single-token de ruido típico."""
    tokens = (text or "").split()
    kept: list[str] = []
    for tok in tokens:
        if _NOISE_SINGLE_TOKEN_RE.match(tok):
            continue
        compact = tok.replace(".", "")
        if compact != tok and _NOISE_SINGLE_TOKEN_RE.match(compact):
            continue
        kept.append(tok)
    return " ".join(kept)


# ============================================================================
# Extractores puros (para filenames / DLNA / rutas)
# ============================================================================


def extract_imdb_id_from_text(text: str) -> str | None:
    """
    Extrae tt1234567 desde un texto cualquiera (case-insensitive).
    Devuelve en lowercase.
    """
    if not text:
        return None
    m = _IMDB_TT_RE.search(text)
    if not m:
        return None
    return m.group(1).lower().strip() or None


def extract_year_from_text(text: str) -> int | None:
    """
    Extrae un año 1900-2099 desde un texto cualquiera.
    - Conservador: devuelve el primer match razonable.
    """
    if not text:
        return None
    m = _YEAR_IN_TEXT_RE.search(text)
    if not m:
        return None
    try:
        y = int(m.group(1))
    except Exception:
        return None
    return y if 1900 <= y <= 2099 else None


def split_title_and_year_from_text(text: str) -> tuple[str, int | None]:
    """
    Intenta separar "Title (1999)" / "Title - 1999" / "Title.1999" -> ("Title", 1999).

    Si no detecta patrón trailing, devuelve (text, year_detected_or_none),
    donde year_detected_or_none puede venir por match “en medio” (conservador).
    """
    raw = (text or "").strip()
    if not raw:
        return "", None

    m = _YEAR_TRAILING_RE.match(raw)
    if m:
        title_part = (m.group("title") or "").strip()
        y_raw = m.group("year1") or m.group("year2") or ""
        try:
            y = int(y_raw)
        except Exception:
            y = None
        return title_part, (y if y is not None and 1900 <= y <= 2099 else None)

    # Fallback: año en cualquier parte (p.ej. "Movie 1999 Remux")
    y2 = extract_year_from_text(raw)
    return raw, y2


def filename_stem(path: str) -> str:
    """
    Devuelve el "stem" del filename sin extensión.
    No toca disco; es puro.
    """
    p = (path or "").strip()
    if not p:
        return ""
    base = os.path.basename(p)
    # si hay múltiples extensiones raras (.mkv.part) se queda con el primer splitext
    stem, _ext = os.path.splitext(base)
    return stem or ""


def clean_title_candidate(text: str) -> str:
    """
    Limpieza conservadora de un candidato a título (p.ej. filename stem):
    - normaliza separadores
    - recorta dash-group ruidoso
    - elimina bracketed-noise (si knob)
    - NO aplica _LOOKUP_NON_ALNUM_RE (eso es para lookup), aquí queremos un “display title” usable.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    t = cleanup_separators(raw)
    t = remove_trailing_dash_group(t)
    t = remove_bracketed_noise(t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


# ============================================================================
# Normalización pública
# ============================================================================


@dataclass(frozen=True)
class NormalizeOptions:
    """
    Opciones para normalización 'compare'.
    - max_len: recorte defensivo para evitar coste en difflib / entradas raras.
    - strip_accents: si quieres comparar sin diacríticos.
    """

    max_len: int | None = None
    strip_accents: bool = False


def normalize_title_for_lookup(title: str) -> str:
    """
    Normalización fuerte para consultas externas (OMDb/Wikipedia).
    """
    raw = (title or "").strip()
    if not raw:
        return ""

    t = cleanup_separators(raw)
    t = remove_trailing_dash_group(t)
    t = remove_bracketed_noise(t)

    if MOVIE_INPUT_LOOKUP_STRIP_ACCENTS:
        t = strip_accents(t)

    t = _LOOKUP_NON_ALNUM_RE.sub(" ", t)
    t = remove_noise_tokens(t)

    t = t.lower().strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t


def normalize_title_for_compare(
    title: str, *, options: NormalizeOptions | None = None
) -> str:
    """
    Normalización conservadora para comparación/identidad:
    - minúsculas
    - recorte defensivo opcional
    - separadores saneados
    - (opcional) strip accents
    - solo [a-z0-9 ] y espacios colapsados
    """
    raw = (title or "").strip()
    if not raw:
        return ""

    opt = options or NormalizeOptions()

    t = cleanup_separators(raw)
    if opt.strip_accents:
        t = strip_accents(t)

    if opt.max_len is not None and opt.max_len > 0 and len(t) > opt.max_len:
        t = t[: opt.max_len]

    t = t.lower()
    t = _COMPARE_NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


__all__ = [
    # separators/cleanup
    "strip_accents",
    "cleanup_separators",
    "remove_bracketed_noise",
    "remove_trailing_dash_group",
    "remove_noise_tokens",
    "looks_like_noise_group",
    "clean_title_candidate",
    "filename_stem",
    # language / script
    "title_has_cjk_script",
    "is_probably_english_title",
    "is_probably_non_english_title",
    "should_skip_title_similarity_due_to_language",
    "generate_sequel_title_variants",
    # extractors
    "extract_imdb_id_from_text",
    "extract_year_from_text",
    "split_title_and_year_from_text",
    # normalization
    "NormalizeOptions",
    "normalize_title_for_lookup",
    "normalize_title_for_compare",
]
