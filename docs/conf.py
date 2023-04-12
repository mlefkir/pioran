# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'pioran'
copyright = '2022, Mehdy Lefkir'
author = 'Mehdy Lefkir'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


extensions = [ 'sphinx.ext.autodoc','sphinx.ext.intersphinx',
                'sphinx.ext.mathjax', 'sphinx.ext.autosummary','sphinx_togglebutton','sphinx.ext.autosectionlabel',
                'sphinx.ext.viewcode',   'numpydoc', 'sphinx.ext.napoleon',
                'myst_nb','sphinx.ext.autodoc.typehints',                 "sphinx_design",
                'sphinx_codeautolink',  
                'sphinx_copybutton','sphinxcontrib.tikz']

intersphinx_mapping = {
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'python': ('https://docs.python.org/3/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None), 
    'equinox': ('https://docs.kidger.site/equinox/', None),
}

autodoc_default_options = {
    'members':          True,
    'undoc-members':    True,
}
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autosectionlabel_prefix_document = True
numpydoc_class_members_toctree = False

# # # Add these lines.
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
# # generate autosummary even if no references

autodoc_default_options = {
    'members':          True,
    'undoc-members':    True,
}
autosummary_generate = True
# autosummary_imported_members = True
# autodoc_inherit_docstrings = True


myst_dmath_allow_labels = True
#'sphinx_autodoc_typehints'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates/autosummary']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','**.ipynb_checkpoints']


latex_elements = {'papersize': 'a4paper',
    'preamble':r'''\usepackage{graphicx}\usepackage{enumitem}\setlistdepth{99}\usepackage{amsmath}\usepackage{amssymb}\def\stackbelow\#1\#2{\underset{\displaystyle\overset{\displaystyle\shortparallel}{\#2}}{\#1}}'''}
tikz_latex_preamble = r'''\usepackage{amsmath}\usepackage{amssymb}'''


#autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# html_theme_options = {
#     'navigation_with_keys': True,
#     "light_css_variables": {
#         "color-brand-primary": "#7C4DFF",
#         "color-brand-content": "#7C4DFF",
#     },
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
