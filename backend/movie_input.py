from __future__ import annotations

"""
MovieInput: Modelo unificado para representar una película independientemente
del origen (Plex, DLNA, fichero local, etc.). Este tipo se utiliza como
entrada estándar del core `analyze_input_movie` y permite desacoplar el
análisis de la capa concreta de datos.

Cumple estrictamente PEP 604, PEP 484, PEP 562/563 y evita el uso de Any.
"""

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Final, Literal


SourceType = Literal["plex", "dlna", "local", "other"]


# ----------------------------------------------------------------------
# Tokens de ruido típicos (release/file)
# ----------------------------------------------------------------------

_NOISE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        480p|576p|720p|1080p|1440p|2160p|4320p|
        4k|8k|uhd|hdr|hdr10|dolby\s*vision|dv|
        x264|x265|h\.?264|h\.?265|hevc|avc|
        bluray|blu-ray|bdrip|brrip|dvdrip|dvd|webdl|web-dl|webrip|hdrip|
        cam|ts|tc|scr|dvdscr|r5|
        proper|repack|remux|limited|unrated|extended|director'?s\s*cut|cut|
        multi|dual(?:\s*audio)?|vose|vos|castellano|espa[nñ]ol|spanish|latino|
        ac3|dts|aac|flac|truehd|atmos|
        subs?|subbed|
        yify|rarbg
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_YEAR_IN_TITLE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s\(\[\-])((?:19|20)\d{2})(?:$|[\s\)\]\-])"
)
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(
    r"[^0-9a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+", re.UNICODE
)
_MULTI_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s{2,}")


# ----------------------------------------------------------------------
# Heurística de idioma (para DLNA/UPnP/local) y soporte Plex (library_language)
# ----------------------------------------------------------------------

# Palabras "muy españolas" o pistas claras
_ES_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \bespa[nñ]ol\b|
        \bcastellano\b|
        \blatino\b|
        \bsubtitulado\b|
        \bdoblada\b|
        \bvose\b|\bvos\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Caracteres/patrones que suelen indicar español (ñ, tildes, etc.)
_ES_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[ñÑáéíóúÁÉÍÓÚüÜ]")

# Artículos/preposiciones muy frecuentes en títulos ES
_ES_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|de|del|y|en|para|con|sin|al)\b",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _cleanup_separators(text: str) -> str:
    t = text.replace("_", " ").replace(".", " ").replace("\u00A0", " ")
    t = t.replace("–", "-").replace("—", "-")
    return t


def _remove_bracketed_noise(text: str) -> str:
    out = text
    patterns: tuple[re.Pattern[str], ...] = (
        re.compile(r"\[[^\]]+\]"),
        re.compile(r"\([^\)]+\)"),
        re.compile(r"\{[^\}]+\}"),
    )

    for pat in patterns:
        while True:
            m = pat.search(out)
            if m is None:
                break
            chunk = out[m.start() : m.end()]
            if _NOISE_TOKEN_RE.search(chunk):
                out = out[: m.start()] + " " + out[m.end() :]
            else:
                inner = chunk[1:-1].strip()
                if not inner or inner.isdigit() or _YEAR_IN_TITLE_RE.search(inner):
                    out = out[: m.start()] + " " + out[m.end() :]
                else:
                    break

    return out


def _remove_trailing_dash_group(text: str) -> str:
    parts = [p.strip() for p in text.split(" - ")]
    if len(parts) <= 1:
        return text

    left = parts[0]
    right = " ".join(parts[1:])
    if _NOISE_TOKEN_RE.search(right):
        return left
    return text


def _remove_noise_tokens(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for tok in tokens:
        if _NOISE_TOKEN_RE.fullmatch(tok):
            continue
        if _NOISE_TOKEN_RE.fullmatch(tok.replace(".", "")):
            continue
        kept.append(tok)
    return " ".join(kept)


def normalize_title_for_lookup(title: str) -> str:
    """
    Normalización centralizada para consultas externas (OMDb/Wikipedia).

    Objetivo:
    - Mantener el título “humano” pero libre de ruido típico de filename/releases.
    - Hacerlo robusto frente a acentos, separadores raros y tokens técnicos.
    """
    raw = title.strip()
    if not raw:
        return ""

    t = _cleanup_separators(raw)
    t = _remove_trailing_dash_group(t)
    t = _remove_bracketed_noise(t)

    # Para búsquedas externas, quitamos acentos para robustez
    t = _strip_accents(t)

    t = _NON_ALNUM_RE.sub(" ", t)
    t = _remove_noise_tokens(t)

    t = t.lower().strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t


def guess_spanish_from_title_or_path(title: str, file_path: str) -> bool:
    """
    Heurística (DLNA/UPnP/local):
    - True si el título o el path contienen pistas claras de español.
    - Conservadora: busca caracteres (ñ/tildes) y palabras funcionales típicas.
    """
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False

    if _ES_HINT_RE.search(haystack):
        return True

    # Caracteres típicos en español
    if _ES_CHAR_RE.search(haystack):
        return True

    # Palabras funcionales comunes en ES: si aparecen varias, suma confianza
    # (evitamos que un "de" aislado dispare)
    words = _cleanup_separators(haystack.lower())
    hits = len(_ES_FUNCTION_WORD_RE.findall(words))
    return hits >= 2


def is_probably_english_title(title: str) -> bool:
    """
    Heurística ligera para detectar si un título "parece inglés".
    No es perfecta, pero sirve para evitar sugerencias obvias.
    """
    t = title.strip()
    if not t:
        return False

    # Si tiene ñ/tildes -> NO es "inglés puro"
    if _ES_CHAR_RE.search(t):
        return False

    # Palabras frecuentes en títulos EN
    en_markers = re.compile(
        r"\b(the|a|an|and|of|to|in|for|with|without|part|chapter|episode)\b",
        re.IGNORECASE,
    )
    return bool(en_markers.search(t))


@dataclass(slots=True)
class MovieInput:
    """
    Representación normalizada de una película antes del análisis.

    - `extra` puede incluir:
        - "library_language": str (Plex) -> ej "es", "es-ES", "spa", "en"
        - otros campos específicos de origen
    """

    source: SourceType
    library: str
    title: str
    year: int | None

    file_path: str
    file_size_bytes: int | None

    imdb_id_hint: str | None
    plex_guid: str | None
    rating_key: str | None
    thumb_url: str | None

    extra: dict[str, object] = field(default_factory=dict)

    def has_physical_file(self) -> bool:
        return bool(self.file_path)

    def normalized_title(self) -> str:
        return self.title.lower().strip()

    def normalized_title_for_lookup(self) -> str:
        return normalize_title_for_lookup(self.title)

    # -------------------------
    # Idioma (Plex y heurística)
    # -------------------------

    def plex_library_language(self) -> str | None:
        """
        Idioma configurado en la librería Plex (si el pipeline Plex lo ha metido en extra).
        Ejemplos: "es", "es-ES", "spa", "en", ...
        """
        val = self.extra.get("library_language")
        return val if isinstance(val, str) and val.strip() else None

    def is_spanish_context(self) -> bool:
        """
        True si:
          - Plex: library_language indica español
          - DLNA/local: heurística por título/path
        """
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith("es") or l.startswith("spa"):
                return True
            return False

        # DLNA/UPnP/local: heurística
        return guess_spanish_from_title_or_path(self.title, self.file_path)

    def describe(self) -> str:
        year_str = str(self.year) if self.year is not None else "?"
        base = f"[{self.source}] {self.title} ({year_str}) / {self.library}"
        if self.file_path:
            base += f" / {self.file_path}"
        return base