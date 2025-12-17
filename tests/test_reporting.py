import json
from pathlib import Path

from backend.reporting import (
    write_all_csv,
    write_filtered_csv,
    write_suggestions_csv,
    write_interactive_html,
)


# ---------------------------------------------------------
# Helper para filas
# ---------------------------------------------------------
def sample_row() -> dict:
    return {
        "poster_url": None,
        "library": "MyLib",
        "title": "Sample Movie",
        "year": 2020,
        "imdb_rating": 7.2,
        "rt_score": 85,
        "imdb_votes": 1200,
        "decision": "MAYBE",
        "reason": "Test",
        "misidentified_hint": "",
        "file": "/path/to/file.mp4",
    }


# ---------------------------------------------------------
# HTML REPORT TESTS
# ---------------------------------------------------------


def test_write_interactive_html_creates_file(tmp_path: Path) -> None:
    rows = [sample_row()]

    out = tmp_path / "report.html"
    write_interactive_html(str(out), rows)

    assert out.exists()
    text = out.read_text(encoding="utf-8")

    # Se han reemplazado los placeholders
    assert "__TITLE__" not in text
    assert "__ROWS_JSON__" not in text

    # Título por defecto presente
    assert "Plex Movies Cleaner" in text

    # Presencia del contenido esperado
    assert "Sample Movie" in text
    assert '<script id="rows-data" type="application/json">' in text

    # Extraemos JSON correctamente
    start = text.find('<script id="rows-data" type="application/json">')
    assert start != -1
    start = text.find(">", start) + 1
    end = text.find("</script>", start)
    rows_json = text[start:end].strip()

    parsed = json.loads(rows_json)
    assert isinstance(parsed, list)
    assert parsed[0]["title"] == "Sample Movie"


def test_write_interactive_html_handles_script_escape(tmp_path: Path) -> None:
    # Caso borde: r['title'] contiene "</script"
    rows = [dict(sample_row(), title="Danger </script tag")]

    out = tmp_path / "escaped.html"
    write_interactive_html(str(out), rows)

    text = out.read_text(encoding="utf-8")

    # Extraemos SOLO el JSON embebido
    start = text.find('<script id="rows-data" type="application/json">')
    assert start != -1
    start = text.find(">", start) + 1
    end = text.find("</script>", start)
    rows_json = text[start:end].strip()

    # Dentro del JSON NO debe aparecer la secuencia literal "</script"
    assert "</script" not in rows_json
    # Pero sí debe haberse escapado como "<\/script"
    assert "<\\/script" in rows_json

    parsed = json.loads(rows_json)
    assert parsed[0]["title"] == "Danger </script tag"


def test_write_interactive_html_custom_title_subtitle(tmp_path: Path) -> None:
    rows = [sample_row()]

    out = tmp_path / "custom.html"
    write_interactive_html(
        str(out),
        rows,
        title="Custom Title",
        subtitle="Custom Subtitle",
    )

    text = out.read_text(encoding="utf-8")
    assert "Custom Title" in text
    assert "Custom Subtitle" in text


def test_write_interactive_html_accepts_iterables(tmp_path: Path) -> None:
    # Iterable en vez de lista
    rows = (r for r in [sample_row()])

    out = tmp_path / "iterable.html"
    write_interactive_html(str(out), rows)

    assert out.exists()


# ---------------------------------------------------------
# CSV WRITING TESTS
# ---------------------------------------------------------


def test_write_all_csv(tmp_path: Path) -> None:
    rows = [sample_row(), dict(sample_row(), title="Another")]
    out = tmp_path / "all.csv"

    write_all_csv(str(out), rows)

    assert out.exists()
    text = out.read_text()
    # cabeceras: deben contener al menos "title"
    assert "title" in text
    assert "Sample Movie" in text
    assert "Another" in text


def test_write_all_csv_empty(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    write_all_csv(str(out), [])
    # No debe explotar, pero tampoco genera un archivo vacío para all.csv
    assert not out.exists()


def test_write_filtered_csv(tmp_path: Path) -> None:
    rows = [
        dict(sample_row(), title="KeepThis", decision="KEEP"),
        dict(sample_row(), title="DeleteThis", decision="DELETE"),
    ]

    out = tmp_path / "filtered.csv"
    write_filtered_csv(str(out), rows)

    assert out.exists()
    text = out.read_text()
    assert "KeepThis" in text
    assert "DeleteThis" in text


def test_write_suggestions_csv_empty(tmp_path: Path) -> None:
    out = tmp_path / "sugg.csv"
    write_suggestions_csv(str(out), [])

    assert out.exists()
    text = out.read_text()

    # Las cabeceras estándar DEBEN aparecer
    assert "plex_guid" in text
    assert "library" in text
    assert "suggestions_json" in text

    # No debe haber filas
    lines = text.strip().splitlines()
    assert len(lines) == 1  # Solo header


def test_write_suggestions_csv_with_rows(tmp_path: Path) -> None:
    row = {
        "plex_guid": "g1",
        "library": "MyLib",
        "plex_title": "Test",
        "plex_year": 2000,
        "omdb_title": "OMDb Test",
        "omdb_year": "2000",
        "imdb_rating": 7.0,
        "imdb_votes": 1000,
        "suggestions_json": '{"new_title":"Test2"}',
    }
    out = tmp_path / "sugg2.csv"

    write_suggestions_csv(str(out), [row])

    assert out.exists()
    text = out.read_text()
    assert "g1" in text
    assert "Test2" in text


# ---------------------------------------------------------
# Robustez ante campos faltantes
# ---------------------------------------------------------


def test_write_interactive_html_missing_optional_fields(tmp_path: Path) -> None:
    # Quitamos campos no esenciales
    row = {
        "title": "NoExtras",
        "library": "Lib",
        "poster_url": None,
    }
    out = tmp_path / "missing_fields.html"

    write_interactive_html(str(out), [row])

    assert out.exists()
    text = out.read_text()
    assert "NoExtras" in text


def test_csv_union_of_keys(tmp_path: Path) -> None:
    # Este test asegura que reporting usa la unión de claves
    row1 = {"a": 1, "b": 2}
    row2 = {"b": 3, "c": 4}

    out = tmp_path / "keys.csv"
    write_all_csv(str(out), [row1, row2])

    text = out.read_text()
    header = text.splitlines()[0]
    # Debe contener columnas a, b, c
    assert "a" in header
    assert "b" in header
    assert "c" in header


def test_write_interactive_html_handles_empty_rows(tmp_path: Path) -> None:
    out = tmp_path / "empty_html.html"

    # Un informe vacío no tiene sentido semántico, pero debe NO fallar.
    write_interactive_html(str(out), [])

    assert out.exists(), "Debe generar el HTML aunque no haya filas."
    text = out.read_text()
    # Debe contener un JSON vacío en la etiqueta correspondiente
    assert '<script id="rows-data" type="application/json">[]</script>' in text