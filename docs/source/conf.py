# docs/source/conf.py

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# -- Project information -----------------------------------------------------

project = "blue-sampler"
author = "Armand de Cacqueray"
copyright = "2026, Armand de Cacqueray"
try:
    from blue_sampler import __version__ as release
except ImportError:
    release = "5.0.1"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = []
exclude_patterns = []

# Generate autosummary pages automatically
autosummary_generate = True

# Sensible defaults for API docs.
# Note: a key present here *enables* the option, whatever its value ("imported-members": False
# would still document imported names), so only list the options that should be on.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Keep type hints in the signature:
# sample_points(N: int, D: int = 2)
autodoc_typehints = "signature"

# Parse Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Docstring preprocessing -------------------------------------------------
# The docstrings are written as plain text: |x| means a norm, not an rST substitution.
# Escape the pipes so that they show up as written instead of breaking the build.

def _escape_pipes(app, what, name, obj, options, lines):
    for i, line in enumerate(lines):
        lines[i] = line.replace("|", r"\|")


def setup(app):
    app.connect("autodoc-process-docstring", _escape_pipes, priority=600)