# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))           # Project root
sys.path.insert(0, os.path.abspath('../../src'))       # src modules

project = 'Bayesian MCMC Framework'
copyright = '2025, Saumik Dana'
author = 'Saumik Dana'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',  # For Jupyter notebook support
]

templates_path = ['_templates']
exclude_patterns = ['**/.ipynb_checkpoints']

# Notebook settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Allow error cells for demonstration

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']