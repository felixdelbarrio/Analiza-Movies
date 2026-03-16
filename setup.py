from pathlib import Path

from setuptools import find_packages, setup

# Load README.md as long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

setup(
    name="analiza-movies",
    version="0.2.1",
    author="Félix del Barrio",
    description=(
        "Desktop-first platform for analyzing Plex and DLNA libraries "
        "with a React frontend and a FastAPI backend."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3.0-only",
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        # Core runtime
        "python-dotenv>=1.0",
        "requests>=2.31",
        "urllib3>=2.0",
        "typing_extensions>=4.9",
        "keyring>=25.6",
        # App runtime
        "plexapi>=4.15",
        "pandas>=2.1",
        "pywebview>=5.4",
        # FastAPI server
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
    ],
    extras_require={
        "dev": [
            # Tooling
            "black==26.3.1",
            "ruff>=0.6",
            "pytest>=8.0",
            "pytest-cov>=4.1",
            # Typing / static analysis
            "mypy>=1.8",
            "pyright>=1.1.408",
            # Stubs
            "pandas-stubs>=2.1",
            "types-requests>=2.31",
            "types-python-dateutil>=2.8",
            # Some pandas-stubs expect these to be importable
            "matplotlib>=3.8",
            "xarray>=2024.1",
            "sqlalchemy>=2.0",
        ],
        "build": [
            "build>=1.2",
            "pyinstaller>=6.11",
        ],
        "viz": [
            "rich>=13.7",
            "plotly>=5.18",
            "sympy>=1.12",
            "graphviz>=0.20",
        ],
        # Si quieres mantener opcionales runtime como "extras" en vez de requirements.txt:
    },
    entry_points={
        "console_scripts": [
            "start=backend.main:start",
            "start-server=server.__main__:main",
            "start-desktop=desktop.app:main",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Environment :: Web Environment",
        "Framework :: FastAPI",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Video",
    ],
)
