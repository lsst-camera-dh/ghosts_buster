import os
import sys
sys.path.insert(0, os.path.abspath('../../src/ghost_buster/'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LSST Ghost Buster'
copyright = '2025, Buffat Dimitri'
author = 'Buffat Dimitri'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Pour les Google/NumPy-style docstrings
    'sphinx.ext.viewcode',  # Lien vers le code source
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["lsst.afw", "pyplot", "photutils"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
