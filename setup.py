from pathlib import Path

from setuptools import find_packages, setup

# Load README.md as long description (if present)
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="analiza-movies",
    version="0.1.0",
    author="Félix del Barrio",
    description=(
        "Toolset for analyzing Plex movie libraries, scoring titles and suggesting deletions "
        "or metadata fixes."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<tu_usuario>/analiza-movies",  # opcional: ajusta o elimina si no aplica
    license="MIT",
    # Packages
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    # Runtime dependencies
    install_requires=[
        "python-dotenv>=1.0",
        "requests>=2.31",
        "plexapi>=4.15",
        "pandas>=2.1",
        "streamlit>=1.32",
        "altair>=5.2",
        "streamlit-aggrid>=0.3.4",
        # Optional but strongly recommended: improves Streamlit hot-reload performance
        "watchdog>=3.0",
    ],
    extras_require={
        "dev": [
            "black",
            "mypy",
            "pyright",
            "pytest",
            "ruff",
        ],
    },
    entry_points={
        "console_scripts": [
            "analiza=backend.analiza:main",
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