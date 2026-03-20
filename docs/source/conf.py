# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'crbdd'
copyright = f'{date.today().year}, Aubrey Moore'
copyright = '2026, Aubrey Moore'
author = 'Aubrey Moore'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.napoleon',   # Support for Google/NumPy docstrings
    'sphinx.ext.viewcode',   # Links to source code
]

# AutoAPI settings: point to your src directory
autoapi_dirs = ['../../src']
autoapi_type = 'python'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
