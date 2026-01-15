"""
backend/movie_input.py

(MovieInput y helpers)

Refactor:
- Mueve utilidades compartidas (normalización, heurísticas básicas y extractores)
  a backend.title_utils para evitar duplicación con decision_logic.py / metadata_fix.py, etc.
- Este módulo se centra en:
  - Modelo MovieInput
  - Heurísticas de contexto (por idioma) basadas en title + path + library language
  - Helpers de "coalescing" para mejorar inputs de DLNA/local (title/year/imdb_id_hint)

Nota:
- Sin logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Final, Literal

from typing_extensions import TypeAlias

from backend.config_plex import (
    MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS,
    MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK,
)
from backend.title_utils import (
    cleanup_separators as _cleanup_separators,
    clean_title_candidate,
    extract_imdb_id_from_text,
    extract_year_from_text,
    filename_stem,
    is_probably_english_title,
    normalize_title_for_lookup,
    split_title_and_year_from_text,
    title_has_cjk_script,
)

# ============================================================================
# Tipos públicos
# ============================================================================

SourceType = Literal["plex", "dlna", "local", "other"]

LanguageCode: TypeAlias = Literal["es", "en", "it", "fr", "ja", "ko", "zh", "unknown"]

# ============================================================================
# Regex/constantes para heurística de idioma (se mantienen aquí)
# (Porque son “contexto” MovieInput, no “core string utils”)
# ============================================================================

# ---------- Español ----------
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
_ES_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[ñÑáéíóúÁÉÍÓÚüÜ]")
_ES_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|de|del|y|en|para|con|sin|al)\b",
    re.IGNORECASE,
)

# ---------- Inglés ----------
_EN_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \benglish\b|\beng\b|\bsubtitles?\b|\bsubbed\b|\bdubbed\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EN_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(the|a|an|and|or|of|to|in|on|for|with|without|from|by|part|chapter|episode)\b",
    re.IGNORECASE,
)

# ---------- Italiano ----------
_IT_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \bitaliano\b|\bitalian\b|\bita\b|\bsottotitol(?:i|ato)\b|\bdoppiat[oa]\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_IT_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(il|lo|la|i|gli|le|un|una|di|del|dello|della|dei|degli|delle|e|in|per|con|senza|al|allo|alla)\b",
    re.IGNORECASE,
)

# ---------- Francés ----------
_FR_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \bfran[cç]ais\b|\bfrench\b|\bvf\b|\bvostfr\b|\bsous-?titres?\b|\bdoubl[ée]e?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_FR_FUNCTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(le|la|les|un|une|des|de|du|et|en|pour|avec|sans|au|aux)\b",
    re.IGNORECASE,
)

# ---------- Japonés ----------
_JA_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \bjapanese\b|\bjpn\b|\bnihongo\b|日本語|字幕|吹替
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------- Coreano ----------
_KO_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \bkorean\b|\bkor\b|한국어|자막|더빙
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------- Chino ----------
_ZH_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \bchinese\b|\bchi\b|\bzho\b|中文|國語|国语|粤语|粵語|字幕|配音
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Unicode blocks (solo para ayudas locales)
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
# Helpers internos (puros, sin logging)
# ============================================================================


def _count_function_word_hits(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _lang_hits_ge(text: str, pattern: re.Pattern[str]) -> bool:
    threshold = int(MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS)
    if threshold <= 0:
        return True
    return _count_function_word_hits(text, pattern) >= threshold


def _best_effort_imdb_hint(title: str, file_path: str) -> str | None:
    """
    Intenta extraer tt1234567 desde (title + file_path).
    Útil en DLNA/local cuando la fuente no trae GUIDs.
    """
    hay = f"{title or ''} {file_path or ''}".strip()
    return extract_imdb_id_from_text(hay) if hay else None


def _best_effort_year(title: str, file_path: str, year: int | None) -> int | None:
    """
    Si year es None, intenta inferirlo de:
    - filename stem (preferido)
    - title
    - ruta completa
    """
    if year is not None:
        return year

    stem = filename_stem(file_path)
    if stem:
        _t, y = split_title_and_year_from_text(stem)
        if y is not None:
            return y
        y2 = extract_year_from_text(stem)
        if y2 is not None:
            return y2

    if title:
        _t2, y3 = split_title_and_year_from_text(title)
        if y3 is not None:
            return y3
        y4 = extract_year_from_text(title)
        if y4 is not None:
            return y4

    if file_path:
        y5 = extract_year_from_text(file_path)
        if y5 is not None:
            return y5

    return None


def _best_effort_title(title: str, file_path: str) -> str:
    """
    Si title viene vacío/raro, intenta derivarlo del filename stem.
    Limpia el candidato de forma conservadora.
    """
    t = (title or "").strip()
    if t:
        return clean_title_candidate(t)

    stem = filename_stem(file_path)
    if stem:
        t2, _y = split_title_and_year_from_text(stem)
        cand = t2.strip() or stem.strip()
        return clean_title_candidate(cand)

    return ""


def coalesce_movie_identity(
    *,
    title: str,
    year: int | None,
    file_path: str,
    imdb_id_hint: str | None,
) -> tuple[str, int | None, str | None]:
    """
    Helper público (puro) para mejorar identidad (title/year/imdb_id_hint) en fuentes pobres (DLNA/local).

    Política:
    1) imdb_id_hint: si ya viene, se respeta; si no, se intenta extraer de title/path.
    2) title: si viene vacío, se deriva de filename stem.
    3) year: si None, se infiere de stem/title/path.
    """
    title2 = _best_effort_title(title, file_path)
    imdb2 = (imdb_id_hint or "").strip() or None
    if imdb2 is None:
        imdb2 = _best_effort_imdb_hint(title2, file_path)

    year2 = _best_effort_year(title2, file_path, year)
    return title2, year2, imdb2


# ============================================================================
# API pública: heurísticas de idioma
# ============================================================================


def guess_spanish_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False

    if _ES_HINT_RE.search(haystack) or _ES_CHAR_RE.search(haystack):
        return True

    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _ES_FUNCTION_WORD_RE)


def guess_english_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False

    if MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK and _CJK_ANY_RE.search(haystack):
        return False

    if _EN_HINT_RE.search(haystack):
        return True

    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _EN_FUNCTION_WORD_RE)


def guess_italian_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    if _IT_HINT_RE.search(haystack):
        return True
    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _IT_FUNCTION_WORD_RE)


def guess_french_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    if _FR_HINT_RE.search(haystack):
        return True
    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _FR_FUNCTION_WORD_RE)


def guess_japanese_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    if _JA_HINT_RE.search(haystack):
        return True
    return bool(_KANA_RE.search(haystack))


def guess_korean_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    return bool(_KO_HINT_RE.search(haystack) or _HANGUL_RE.search(haystack))


def guess_chinese_from_title_or_path(title: str, file_path: str) -> bool:
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False

    if _ZH_HINT_RE.search(haystack):
        return True

    if (
        _HAN_RE.search(haystack)
        and not _KANA_RE.search(haystack)
        and not _HANGUL_RE.search(haystack)
    ):
        return True

    return False


def detect_context_language_code(movie_input: "MovieInput") -> LanguageCode:
    lang = movie_input.plex_library_language()
    if lang:
        lang_code = lang.strip().lower()
        if lang_code.startswith(("es", "spa")):
            return "es"
        if lang_code.startswith(("en", "eng")):
            return "en"
        if lang_code.startswith(("it", "ita")):
            return "it"
        if lang_code.startswith(("fr", "fra", "fre")):
            return "fr"
        if lang_code.startswith(("ja", "jp", "jpn")):
            return "ja"
        if lang_code.startswith(("ko", "kor")):
            return "ko"
        if lang_code.startswith(("zh", "chi", "zho")):
            return "zh"

    if movie_input.is_japanese_context():
        return "ja"
    if movie_input.is_korean_context():
        return "ko"
    if movie_input.is_chinese_context():
        return "zh"
    if movie_input.is_spanish_context():
        return "es"
    if movie_input.is_italian_context():
        return "it"
    if movie_input.is_french_context():
        return "fr"
    if movie_input.is_english_context():
        return "en"

    return "unknown"


def should_skip_new_title_suggestion(
    *,
    context_lang: LanguageCode,
    current_title: str,
    omdb_title: str,
) -> bool:
    cur = (current_title or "").strip()
    om = (omdb_title or "").strip()
    if not om:
        return False

    if not is_probably_english_title(om):
        return False

    if context_lang in ("en", "unknown"):
        return False

    current_is_english = is_probably_english_title(cur)

    if context_lang in ("ja", "ko", "zh"):
        if title_has_cjk_script(cur) and not current_is_english:
            return True
        return False

    if context_lang in ("es", "it", "fr"):
        return not current_is_english

    return False


# ============================================================================
# Modelo unificado
# ============================================================================


@dataclass
class MovieInput:
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

    # ----------------------------
    # Helpers básicos
    # ----------------------------

    def has_physical_file(self) -> bool:
        return bool((self.file_path or "").strip())

    def normalized_title(self) -> str:
        return (self.title or "").lower().strip()

    def normalized_title_for_lookup(self) -> str:
        return normalize_title_for_lookup(self.title or "")

    def plex_library_language(self) -> str | None:
        val = self.extra.get("library_language")
        if isinstance(val, str):
            v = val.strip()
            return v or None
        return None

    # ----------------------------
    # Context language (best-effort)
    # ----------------------------

    def is_spanish_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("es", "spa")):
                return True
        return guess_spanish_from_title_or_path(self.title or "", self.file_path or "")

    def is_english_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("en", "eng")):
                return True
        return guess_english_from_title_or_path(self.title or "", self.file_path or "")

    def is_italian_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("it", "ita")):
                return True
        return guess_italian_from_title_or_path(self.title or "", self.file_path or "")

    def is_french_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("fr", "fra", "fre")):
                return True
        return guess_french_from_title_or_path(self.title or "", self.file_path or "")

    def is_japanese_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("ja", "jp", "jpn")):
                return True
        return guess_japanese_from_title_or_path(self.title or "", self.file_path or "")

    def is_korean_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("ko", "kor")):
                return True
        return guess_korean_from_title_or_path(self.title or "", self.file_path or "")

    def is_chinese_context(self) -> bool:
        lang = self.plex_library_language()
        if lang:
            lang_code = lang.lower().strip()
            if lang_code.startswith(("zh", "chi", "zho")):
                return True
        return guess_chinese_from_title_or_path(self.title or "", self.file_path or "")

    # ----------------------------
    # Identity enrichment (DLNA/local)
    # ----------------------------

    def enriched_identity(self) -> tuple[str, int | None, str | None]:
        """
        Devuelve (title, year, imdb_id_hint) con best-effort.
        No muta el objeto.
        """
        return coalesce_movie_identity(
            title=self.title or "",
            year=self.year,
            file_path=self.file_path or "",
            imdb_id_hint=self.imdb_id_hint,
        )

    # ----------------------------
    # Describe
    # ----------------------------

    def describe(self) -> str:
        year_str = str(self.year) if self.year is not None else "?"
        base = f"[{self.source}] {self.title} ({year_str}) / {self.library}"
        fp = (self.file_path or "").strip()
        if fp:
            base += f" / {fp}"
        return base


__all__ = [
    "MovieInput",
    "SourceType",
    "LanguageCode",
    "detect_context_language_code",
    "should_skip_new_title_suggestion",
    "guess_spanish_from_title_or_path",
    "guess_english_from_title_or_path",
    "guess_italian_from_title_or_path",
    "guess_french_from_title_or_path",
    "guess_japanese_from_title_or_path",
    "guess_korean_from_title_or_path",
    "guess_chinese_from_title_or_path",
    "coalesce_movie_identity",
]
