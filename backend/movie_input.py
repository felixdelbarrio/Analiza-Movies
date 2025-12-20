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

_YEAR_IN_TITLE_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|[\s\(\[\-])((?:19|20)\d{2})(?:$|[\s\)\]\-])")
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+", re.UNICODE)
_MULTI_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s{2,}")


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
                # Si el chunk es solo numérico/año o casi vacío, también lo eliminamos
                inner = chunk[1:-1].strip()
                if not inner or inner.isdigit() or _YEAR_IN_TITLE_RE.search(inner):
                    out = out[: m.start()] + " " + out[m.end() :]
                else:
                    break

    return out


def _remove_trailing_dash_group(text: str) -> str:
    # Casos típicos: "Titulo - 1080p x264 ..." / "Titulo - YIFY"
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
        # tokens con puntos (x264, h.264) etc.
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

    # Convertimos acentos -> base para mejorar búsquedas cuando el origen no coincide
    t = _strip_accents(t)

    # Sustituimos puntuación por espacios, pero preservamos letras/números
    t = _NON_ALNUM_RE.sub(" ", t)

    t = _remove_noise_tokens(t)

    t = t.lower().strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t


@dataclass(slots=True)
class MovieInput:
    """
    Representación normalizada de una película antes del análisis.

    - `source`: origen ("plex", "dlna", "local", "other").
    - `library`: nombre de la biblioteca o categoría.
    - `title`: título a analizar / consultar.
    - `year`: año de lanzamiento (si disponible).
    - `file_path`: ruta del fichero físico (si existe). Obligatoria aunque sea "".
    - `file_size_bytes`: tamaño en bytes si se conoce.
    - `imdb_id_hint`: posible ID de IMDb detectado (puede venir de Plex).
    - `plex_guid`: GUID propio de Plex si existe (None para DLNA/local).
    - `rating_key`: clave interna de Plex para reproducir o identificar elementos.
    - `thumb_url`: miniatura cuando la plataforma la proporciona.
    - `extra`: diccionario extensible para datos específicos del origen.

    Este objeto es estable, ligero y seguro para ser manipulado y analizado.
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

    # ----------------------------------------------------------------------
    # Métodos auxiliares (opcionales pero útiles)
    # ----------------------------------------------------------------------

    def has_physical_file(self) -> bool:
        """Devuelve True si existe una ruta de fichero válida."""
        return bool(self.file_path)

    def normalized_title(self) -> str:
        """Devuelve un título en minúsculas para búsquedas insensibles."""
        return self.title.lower().strip()

    def normalized_title_for_lookup(self) -> str:
        """Devuelve el título depurado centralmente para búsquedas externas."""
        return normalize_title_for_lookup(self.title)

    def describe(self) -> str:
        """
        Devuelve una cadena útil para logs.
        Ejemplo: "[plex] Matrix (1999) / Movies / file.mkv"
        """
        year_str = str(self.year) if self.year is not None else "?"
        base = f"[{self.source}] {self.title} ({year_str}) / {self.library}"
        if self.file_path:
            base += f" / {self.file_path}"
        return base