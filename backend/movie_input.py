from __future__ import annotations

"""
MovieInput: Modelo unificado para representar una película independientemente
del origen (Plex, DLNA, fichero local, etc.). Este tipo se utiliza como
entrada estándar del core `analyze_input_movie` y permite desacoplar el
análisis de la capa concreta de datos.

Cumple estrictamente PEP 604, PEP 484, PEP 562/563 y evita el uso de Any.
"""

from dataclasses import dataclass, field
from typing import Literal


SourceType = Literal["plex", "dlna", "local", "other"]


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