# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pioran'
copyright = '2023, Mehdy Lefkir'
author = 'Mehdy Lefkir'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['numpydoc','sphinx.ext.autodoc.typehints','sphinx_codeautolink',
              'sphinx_copybutton','sphinx.ext.mathjax','myst_nb','sphinxcontrib.tikz',
              'sphinx.ext.viewcode','autoapi.extension','sphinx.ext.intersphinx',]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'python': ('https://docs.python.org/3/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None), 
    'equinox': ('https://docs.kidger.site/equinox/', None),
    'tinygp': ('https://tinygp.readthedocs.io/en/latest/', None)
}

autoapi_dirs = ['../src/pioran']
autoapi_type = "python"
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members","special-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members" ]
    
autoapi_template_dir = "_templates/autoapi"
autoapi_python_use_implicit_namespaces = True

myst_dmath_allow_labels = True


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

latex_elements = {'papersize': 'a4paper',
    'preamble':r'''\usepackage{graphicx}
    \usepackage{enumitem}
    \setlistdepth{99}
    \usepackage{amsmath}\usepackage{amssymb}''',
    'figure_align': 'htbp',
}
tikz_latex_preamble = r'''\usepackage{amsmath}\usepackage{amssymb}'''

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
html_theme = 'pydata_sphinx_theme'
html_title = "pioran"
html_short_title = "pioran"
html_logo = '_static/logo_borderbis.png'
latex_logo = '_static/logo_bg.pdf'
html_favicon = '_static/favico.ico'
html_css_files = [
    "css/custom.css",
]
# html_theme_options = {
#     'navigation_with_keys': True,
#     "light_css_variables": {
#         "color-brand-primary": "#0077bb",
#         "color-brand-content": "#0077bb",
#     },
# }
html_static_path = ['_static']


# https://bylr.info/articles/2022/05/10/api-doc-with-sphinx-autoapi/#autoapi-objects
rst_prolog = """
.. role:: summarylabel
"""

def contains(seq, item):
    return item in seq


def prepare_jinja_env(jinja_env) -> None:
    jinja_env.tests["contains"] = contains

autoapi_prepare_jinja_env = prepare_jinja_env
