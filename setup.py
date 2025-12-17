from setuptools import setup, find_packages
from pathlib import Path

# Load README.md as long description (if present)
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="analiza-movies",
    version="0.1.0",
    author="Félix del Barrio",
    description="Toolset for analyzing Plex movie libraries, scoring titles and suggesting deletions or metadata fixes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<tu_usuario>/analiza-movies",  # opcional: puedes ajustarlo o quitarlo
    license="MIT",

    # Packages
    packages=find_packages(),
    include_package_data=True,  # por si decides incluir CSV/json futuros vía MANIFEST.in

    # Runtime dependencies
    install_requires=[
        "python-dotenv",
        "requests",
        "plexapi",
        "pandas",
        "streamlit",
        "altair",
        "streamlit-aggrid",
    ],

    # Optional dependencies (developer tools, etc.)
    extras_require={
        "dev": [
            "black",
            "mypy",
            "pyright",
            "pytest",
            "ruff",
        ],
    },

    # Entry points: permite lanzar comandos instalables
    entry_points={
        "console_scripts": [
            # Ejecuta el análisis interactivo (elige Plex o DLNA)
            "analiza=analiza:main",
            # Ejecuta análisis Plex directamente (todas las bibliotecas)
            "analiza-plex=backend.analiza_plex:analyze_all_libraries",
            # Ejecuta análisis DLNA/directorio local directamente
            "analiza-dlna=backend.analiza_dlna:analyze_dlna_server",
            # Si quieres un comando para lanzar el dashboard en el futuro:
            # "analiza-dashboard=backend.dashboard:run_dashboard",
        ]
    },

    python_requires=">=3.9",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Video",
        "Topic :: Utilities",
    ],

    keywords="plex movies streamlit analysis metadata cleanup",
)