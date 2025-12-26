from __future__ import annotations

"""
backend/movie_input.py

MovieInput: modelo unificado para representar una película independientemente del origen
(Plex, DLNA, fichero local, etc.).

Este tipo se usa como entrada estándar del core de análisis (p.ej. analyze_movie /
analyze_input_movie) y permite desacoplar:
- descubrimiento / extracción (Plex/DLNA/local)
- análisis (OMDb/Wiki/scoring/misidentified)
- reporting (CSVs)

Principios (alineados con el proyecto)
--------------------------------------
- Tipado estricto (PEP 484 / PEP 604) y sin Any.
- Helpers puros y testables para normalización y heurística.
- Normalización “amigable” para búsquedas externas:
    * limpia tokens típicos de releases/filenames (resolución, codecs, tags...)
    * normaliza separadores raros (., _, NBSP, guiones largos)
    * (configurable) elimina acentos (robustez)
    * colapsa espacios
- Heurística de idioma conservadora:
    * NO pretende ser un detector perfecto; solo ayuda a decisiones (metadata_fix, wiki).
    * preferimos "unknown" antes que asignar mal.
- Este módulo NO hace logging: es un modelo/utility “core” y debe ser silencioso.

Centralización de lógica de idioma
----------------------------------
Para evitar duplicación (p. ej. en metadata_fix.py o wiki_client.py), este módulo expone:

- LanguageCode (TypeAlias)
- detect_context_language_code(movie_input)
- title_has_cjk_script(title)
- is_probably_english_title(title)
- should_skip_new_title_suggestion(context_lang, current_title, omdb_title)

De este modo:
- las heurísticas viven aquí
- la política de uso vive en módulos superiores (p.ej. metadata_fix.py)

Integración con config.py
-------------------------
Este módulo lee "knobs" definidos en backend/config.py para modular comportamiento
sin tocar código:

Lookup (normalización para búsquedas):
- MOVIE_INPUT_LOOKUP_STRIP_ACCENTS
- MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE
- MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP

Heurística idioma:
- MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS
- MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK

⚠️ Nota de arquitectura:
- movie_input.py sigue siendo “silencioso” (sin logs), aunque importe config.
- Si quieres que sea 100% libre de config, podríamos inyectar los flags desde arriba;
  por ahora se prioriza consistencia y centralización de política.

Notas de diseño
---------------
- normalize_title_for_lookup() se usa para consultas externas (OMDb/Wikipedia).
- MovieInput.normalized_title() es ligera y NO limpia ruido.
- extra es dict[str, object] para flexibilidad controlada sin introducir Any.
"""

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Final, Literal, TypeAlias

from backend.config_plex import (
    MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS,
    MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK,
    MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE,
    MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP,
    MOVIE_INPUT_LOOKUP_STRIP_ACCENTS,
)

# ============================================================================
# Tipos públicos
# ============================================================================

SourceType = Literal["plex", "dlna", "local", "other"]

LanguageCode: TypeAlias = Literal["es", "en", "it", "fr", "ja", "ko", "zh", "unknown"]

# ============================================================================
# Regex/constantes para normalización (lookup)
# ============================================================================

# Año plausible (para limpiar "Title (1999)" / "[1999]" / etc.)
_YEAR_IN_TITLE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s\(\[\-])((?:19|20)\d{2})(?:$|[\s\)\]\-])"
)

# Para “sanitizar” a tokens comparables (conservando letras/dígitos y algunos acentos
# para no destrozar demasiado antes de (opcionalmente) quitar diacríticos).
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(
    r"[^0-9A-Za-záéíóúÁÉÍÓÚñÑüÜ]+", re.UNICODE
)
_MULTI_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s{2,}")

# Separadores típicos de releases
_SEP_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("_", " "),
    (".", " "),
    ("\u00A0", " "),  # NBSP
    ("–", "-"),
    ("—", "-"),
)

# ============================================================================
# “Noise” tokens: releases/filenames (codec, rip, edition, audio, subs, tags)
# ----------------------------------------------------------------------------
# Importante: estos patterns están pensados para ELIMINAR “ruido” en normalización
# de lookup. Deben ser conservadores para no destruir el título real.
# ============================================================================

# Tokens típicos “de una palabra” que queremos filtrar (comparando por token)
_NOISE_SINGLE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^(?:
        # resoluciones / formatos
        480p|576p|720p|1080p|1440p|2160p|4320p|
        4k|8k|uhd|hdr|hdr10|dv|

        # codecs
        x264|x265|h\.?264|h\.?265|hevc|avc|

        # sources / rips
        bluray|blu-?ray|bdrip|brrip|dvdrip|dvd|web-?dl|webrip|hdrip|
        cam|ts|tc|scr|dvdscr|r5|

        # editions / tags
        proper|repack|remux|limited|unrated|extended|cut|

        # idiomas / audio/subs (tokens compactos)
        multi|dual|vose|vos|
        castellano|espa[nñ]ol|spanish|latino|
        eng|english|
        ita|italian|italiano|
        fra|fre|french|fran[cç]ais|
        jpn|jap|japanese|nihongo|
        kor|korean|
        chi|zho|chinese|

        # audio
        ac3|dts|aac|flac|truehd|atmos|

        # subs
        subs?|subbed|

        # groups/tags
        yify|rarbg
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Frases típicas “multi-token” que pueden venir en paréntesis/grupos y queremos detectar
# como ruido (p.ej. "Dolby Vision", "Director's Cut", "Dual Audio").
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
# Heurística de idioma (tokens + unicode + function words)
# ----------------------------------------------------------------------------
# Filosofía: conservadora, preferimos "unknown" antes que asignar mal.
#
# Importante: los umbrales “numéricos” (function words) se configuran en config.py
# (MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS).
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

# Unicode blocks (pistas fuertes por escritura; no equivalen 1:1 a idioma)
_KANA_RE: Final[re.Pattern[str]] = re.compile(r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F]")
_HANGUL_RE: Final[re.Pattern[str]] = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
_HAN_RE: Final[re.Pattern[str]] = re.compile(r"[\u4E00-\u9FFF]")
_CJK_ANY_RE: Final[re.Pattern[str]] = re.compile(
    r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F\u4E00-\u9FFF\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]"
)

# ============================================================================
# Helpers internos (puros, sin logging)
# ============================================================================


def _strip_accents(text: str) -> str:
    """
    Elimina diacríticos para robustez en búsquedas externas.

    Controlado por MOVIE_INPUT_LOOKUP_STRIP_ACCENTS.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _cleanup_separators(text: str) -> str:
    """Normaliza separadores típicos de filenames y algunos guiones unicode."""
    out = text
    for a, b in _SEP_REPLACEMENTS:
        out = out.replace(a, b)
    return out


def _looks_like_noise_group(text: str) -> bool:
    """
    Decide si un bloque entre [](){} parece “ruido” técnico.

    Señales:
    - contiene tokens single-token de ruido
    - o contiene frases de ruido
    - o es básicamente numérico/año

    Nota:
    - Conservador: ante dudas, preferimos NO eliminar.
    """
    t = text.strip()
    if not t:
        return True

    if _NOISE_PHRASE_RE.search(t):
        return True

    toks = _cleanup_separators(t).split()
    for tok in toks:
        if _NOISE_SINGLE_TOKEN_RE.match(tok):
            return True

    if t.isdigit():
        return True
    if _YEAR_IN_TITLE_RE.search(t):
        return True

    return False


def _remove_bracketed_noise(text: str) -> str:
    """
    Elimina grupos entre [] () {} cuando parecen ruido.

    Controlado por MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE.

    Conservador:
    - solo elimina el grupo si el contenido “parece ruido” (ver _looks_like_noise_group)
    - si no, deja el texto intacto.
    """
    if not MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE:
        return text

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
            inner = out[m.start() + 1 : m.end() - 1]
            if _looks_like_noise_group(inner):
                out = out[: m.start()] + " " + out[m.end() :]
                continue
            break

    return out


def _remove_trailing_dash_group(text: str) -> str:
    """
    Si el título es "Movie - 1080p - x265" y lo de la derecha parece ruido, recortamos.

    Controlado por MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP.

    Regla:
    - si el segmento a la derecha contiene señales claras de ruido, nos quedamos con la izquierda.
    """
    if not MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP:
        return text

    # split exacto por " - " (muy común en nombres) para evitar cortar guiones internos.
    parts = [p.strip() for p in text.split(" - ")]
    if len(parts) <= 1:
        return text

    left = parts[0].strip()
    if not left:
        return text

    right = " ".join(parts[1:]).strip()
    if _looks_like_noise_group(right):
        return left

    return text


def _remove_noise_tokens(text: str) -> str:
    """
    Elimina tokens que sean ruido típico.

    Nota:
    - Se compara por token completo.
    - Se intenta también contra una versión “compacta” sin puntos (h.264 -> h264).
    """
    tokens = text.split()
    kept: list[str] = []
    for tok in tokens:
        if _NOISE_SINGLE_TOKEN_RE.match(tok):
            continue
        compact = tok.replace(".", "")
        if compact != tok and _NOISE_SINGLE_TOKEN_RE.match(compact):
            continue
        kept.append(tok)
    return " ".join(kept)


def _count_function_word_hits(text: str, pattern: re.Pattern[str]) -> int:
    """Cuenta hits de function words; el caller decide umbrales."""
    return len(pattern.findall(text))


def _lang_hits_ge(text: str, pattern: re.Pattern[str]) -> bool:
    """
    Helper común para evaluar function words con umbral configurable.

    MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS:
    - 0 => “siempre True” (ojo: solo recomendado para debugging)
    - 1/2/... => umbral conservador
    """
    threshold = int(MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS)
    if threshold <= 0:
        return True
    return _count_function_word_hits(text, pattern) >= threshold


# ============================================================================
# API pública: normalización / heurísticas
# ============================================================================


def normalize_title_for_lookup(title: str) -> str:
    """
    Normalización fuerte para consultas externas (OMDb/Wikipedia).

    Propiedades:
    - determinista
    - robusta a separadores / acentos (configurable)
    - minimiza ruido de releases (resolución, codecs, tags...)
    - devuelve una clave en minúsculas con espacios colapsados

    Importante:
    - Esta función NO intenta “traducir” ni “localizar” títulos.
    - Está pensada para maximizar el matching en índices externos.
    """
    raw = (title or "").strip()
    if not raw:
        return ""

    t = _cleanup_separators(raw)
    t = _remove_trailing_dash_group(t)
    t = _remove_bracketed_noise(t)

    # Para lookup: quitamos acentos/diacríticos (si está habilitado)
    if MOVIE_INPUT_LOOKUP_STRIP_ACCENTS:
        t = _strip_accents(t)

    # Solo dejamos letras/dígitos (y luego tokenizamos)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _remove_noise_tokens(t)

    t = t.lower().strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t


def guess_spanish_from_title_or_path(title: str, file_path: str) -> bool:
    """Heurística conservadora de “contexto español”."""
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False

    if _ES_HINT_RE.search(haystack) or _ES_CHAR_RE.search(haystack):
        return True

    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _ES_FUNCTION_WORD_RE)


def guess_english_from_title_or_path(title: str, file_path: str) -> bool:
    """
    Heurística conservadora de “contexto inglés”.

    Control:
    - MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK:
        True  -> si hay CJK/Hangul, devolvemos False (evita falsos positivos).
        False -> mantiene heurística normal.
    """
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
    """Heurística conservadora de “contexto italiano”."""
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    if _IT_HINT_RE.search(haystack):
        return True
    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _IT_FUNCTION_WORD_RE)


def guess_french_from_title_or_path(title: str, file_path: str) -> bool:
    """Heurística conservadora de “contexto francés”."""
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    if _FR_HINT_RE.search(haystack):
        return True
    words = _cleanup_separators(haystack.lower())
    return _lang_hits_ge(words, _FR_FUNCTION_WORD_RE)


def guess_japanese_from_title_or_path(title: str, file_path: str) -> bool:
    """
    Heurística conservadora de “contexto japonés”:
    - tokens (日本語/nihongo/jpn/japanese/字幕/吹替)
    - o presencia de Kana (señal muy fuerte)
    """
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    if _JA_HINT_RE.search(haystack):
        return True
    return bool(_KANA_RE.search(haystack))


def guess_korean_from_title_or_path(title: str, file_path: str) -> bool:
    """
    Heurística conservadora de “contexto coreano”:
    - tokens (한국어/kor/korean/자막/더빙)
    - o presencia de Hangul (señal muy fuerte)
    """
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False
    return bool(_KO_HINT_RE.search(haystack) or _HANGUL_RE.search(haystack))


def guess_chinese_from_title_or_path(title: str, file_path: str) -> bool:
    """
    Heurística conservadora de “contexto chino”:
    - tokens (中文/国语/國語/粤语/粵語/字幕/配音/chi/zho/chinese)
    - o presencia de Han (ideogramas) sin Kana ni Hangul.
    """
    haystack = f"{title} {file_path}".strip()
    if not haystack:
        return False

    if _ZH_HINT_RE.search(haystack):
        return True

    if _HAN_RE.search(haystack) and not _KANA_RE.search(haystack) and not _HANGUL_RE.search(haystack):
        return True

    return False


def title_has_cjk_script(title: str) -> bool:
    """
    True si el texto contiene escritura CJK/Hangul.

    Importante:
    - Esto NO determina el idioma exacto.
    - Se usa como señal fuerte de “título no-latino”.
    """
    return bool(title and _CJK_ANY_RE.search(title))


def is_probably_english_title(title: str) -> bool:
    """
    Heurística ligera: True si el título parece inglés.

    Reglas:
    - Si contiene tildes/ñ -> no lo consideramos “inglés puro”.
    - Si contiene escritura CJK/Hangul -> no lo consideramos inglés.
    - Si contiene function words EN -> “probablemente inglés”.

    Nota:
    - Esta función es intencionadamente simple; se usa para decidir si "proteger"
      títulos locales frente a sugerencias de OMDb en inglés.
    """
    t = (title or "").strip()
    if not t:
        return False
    if _ES_CHAR_RE.search(t):
        return False
    if _CJK_ANY_RE.search(t):
        return False
    return bool(_EN_FUNCTION_WORD_RE.search(_cleanup_separators(t.lower())))


def detect_context_language_code(movie_input: "MovieInput") -> LanguageCode:
    """
    Determina el idioma “de contexto” del ítem.

    Prioridad:
    1) Plex: movie_input.plex_library_language() (si existe)
    2) Heurísticas por título/path encapsuladas en MovieInput (conservadoras)

    Nota:
    - "unknown" es un resultado válido y preferible a “inventar”.
    """
    lang = movie_input.plex_library_language()
    if lang:
        l = lang.strip().lower()
        # Mapeo ISO aproximado
        if l.startswith(("es", "spa")):
            return "es"
        if l.startswith(("en", "eng")):
            return "en"
        if l.startswith(("it", "ita")):
            return "it"
        if l.startswith(("fr", "fra", "fre")):
            return "fr"
        if l.startswith(("ja", "jp", "jpn")):
            return "ja"
        if l.startswith(("ko", "kor")):
            return "ko"
        if l.startswith(("zh", "chi", "zho")):
            return "zh"

    # Fallback: heurísticas del modelo (orden conservador: scripts fuertes primero)
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
    """
    Decide si debemos BLOQUEAR la sugerencia de new_title por reglas multi-idioma.

    Bloquea SOLO cuando:
    - OMDb parece inglés
    - y el contexto es localizable (ES/IT/FR/JA/KO/ZH)
    - y el título actual NO parece inglés
      (en JA/KO/ZH además protegemos si el título actual contiene escritura CJK)

    Uso típico:
      ctx = detect_context_language_code(movie_input)
      if should_skip_new_title_suggestion(ctx, plex_title, omdb_title): ...
    """
    cur = (current_title or "").strip()
    om = (omdb_title or "").strip()
    if not om:
        return False

    # Si OMDb no parece inglés, no aplicamos esta regla.
    if not is_probably_english_title(om):
        return False

    # Contextos no-localizables -> no bloqueamos
    if context_lang in ("en", "unknown"):
        return False

    current_is_english = is_probably_english_title(cur)

    # CJK contexts: protegemos títulos con escritura CJK si no parecen inglés
    if context_lang in ("ja", "ko", "zh"):
        if title_has_cjk_script(cur) and not current_is_english:
            return True
        return False

    # ES/IT/FR: si el título actual no parece inglés, lo protegemos
    if context_lang in ("es", "it", "fr"):
        return not current_is_english

    return False


# ============================================================================
# Modelo unificado
# ============================================================================


@dataclass(slots=True)
class MovieInput:
    """
    Representación unificada de una película antes del análisis.

    Importante:
    - Este modelo NO garantiza que file_path exista (no hay I/O aquí).
    - `extra` permite adjuntar señales del origen (p.ej. display_title, library_language).
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

    # -------------------------
    # Helpers de uso común
    # -------------------------

    def has_physical_file(self) -> bool:
        """True si hay un file_path no vacío (sin verificar existencia)."""
        return bool((self.file_path or "").strip())

    def normalized_title(self) -> str:
        """
        Normalización ligera local: minúsculas + strip (sin limpiar ruido).

        Útil para:
        - comparaciones internas rápidas
        - claves temporales de caches en capas superiores (cuando se quiere conservar ruido)
        """
        return (self.title or "").lower().strip()

    def normalized_title_for_lookup(self) -> str:
        """
        Normalización fuerte para búsquedas externas (OMDb/Wikipedia).

        Respeta flags de config:
        - MOVIE_INPUT_LOOKUP_STRIP_ACCENTS
        - MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE
        - MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP
        """
        return normalize_title_for_lookup(self.title or "")

    # -------------------------
    # Idioma (Plex y heurística)
    # -------------------------

    def plex_library_language(self) -> str | None:
        """
        Idioma configurado/inferido para la librería Plex (si el pipeline lo inyecta en extra).

        Ejemplos válidos:
        - "es", "es-ES", "spa"
        - "en", "en-US", "eng"
        - "it", "ita"
        - "fr", "fra"
        - "ja", "jpn"
        - "ko", "kor"
        - "zh", "zho"
        """
        val = self.extra.get("library_language")
        if isinstance(val, str):
            v = val.strip()
            return v or None
        return None

    def is_spanish_context(self) -> bool:
        """True si library_language indica ES o la heurística por título/path lo sugiere."""
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("es", "spa")):
                return True
        return guess_spanish_from_title_or_path(self.title or "", self.file_path or "")

    def is_english_context(self) -> bool:
        """
        True si library_language indica EN o heurística sugiere inglés.

        Nota:
        - La heurística puede bloquear EN si MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK=True
          y se detecta escritura CJK/Hangul.
        """
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("en", "eng")):
                return True
        return guess_english_from_title_or_path(self.title or "", self.file_path or "")

    def is_italian_context(self) -> bool:
        """True si library_language indica IT o heurística sugiere italiano."""
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("it", "ita")):
                return True
        return guess_italian_from_title_or_path(self.title or "", self.file_path or "")

    def is_french_context(self) -> bool:
        """True si library_language indica FR o heurística sugiere francés."""
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("fr", "fra", "fre")):
                return True
        return guess_french_from_title_or_path(self.title or "", self.file_path or "")

    def is_japanese_context(self) -> bool:
        """True si library_language indica JA/JP o heurística sugiere japonés."""
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("ja", "jp", "jpn")):
                return True
        return guess_japanese_from_title_or_path(self.title or "", self.file_path or "")

    def is_korean_context(self) -> bool:
        """True si library_language indica KO o heurística sugiere coreano."""
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("ko", "kor")):
                return True
        return guess_korean_from_title_or_path(self.title or "", self.file_path or "")

    def is_chinese_context(self) -> bool:
        """True si library_language indica ZH/CHI/ZHO o heurística sugiere chino."""
        lang = self.plex_library_language()
        if lang:
            l = lang.lower().strip()
            if l.startswith(("zh", "chi", "zho")):
                return True
        return guess_chinese_from_title_or_path(self.title or "", self.file_path or "")

    # -------------------------
    # Utilidad para logs/trazas (sin emitir logs aquí)
    # -------------------------

    def describe(self) -> str:
        """
        Describe el item de forma corta para logs/trazas en capas superiores.

        Nota:
        - Este método NO hace logging, solo devuelve un string.
        """
        year_str = str(self.year) if self.year is not None else "?"
        base = f"[{self.source}] {self.title} ({year_str}) / {self.library}"
        fp = (self.file_path or "").strip()
        if fp:
            base += f" / {fp}"
        return base