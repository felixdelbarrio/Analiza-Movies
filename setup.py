from pathlib import Path
from setuptools import find_packages, setup

# Load README.md as long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8")
    if readme_path.exists()
    else ""
)

setup(
    name="analiza-movies",
    version="0.2.1",
    author="Félix del Barrio",
    description=(
        "Toolset for analyzing Plex movie libraries, exposing reports "
        "via Streamlit and a FastAPI backend."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[
        "python-dotenv>=1.0",
        "requests>=2.31",
        "plexapi>=4.15",
        "pandas>=2.1",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "streamlit>=1.32",
        "altair>=5.2",
        "streamlit-aggrid>=0.3.4",
        "watchdog>=3.0",
    ],
    extras_require={
        "dev": [
            "black",
            "ruff",
            "pytest",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "start=backend.main:start",
            # ⬇️ CAMBIO CLAVE
            "start-server=server.__main__:main",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Framework :: FastAPI",
        "Framework :: Streamlit",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Video",
    ],
)