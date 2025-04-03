from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath('../'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QC Lab'
year = datetime.now().year
copyright = f'2025-{year}, Tempelaar Team'
author = 'Tempelaar Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#import sphinx_rtd_theme
#html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'pydata_sphinx_theme' #'bizstyle'#
html_static_path = ['_static']
html_theme_options = {
    'body_max_width': '100%',  # Remove maximum width constraint
    "github_url": "https://github.com/tempelaar-team/qc_lab",
    "collapse_navigation": True,
    "show_nav_level":2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "external_links_new_tab": True,
    "logo": {
        "image_light": "_static/images/logo-light.png",
        "image_dark": "_static/images/logo-dark.png",
    }
}
html_css_files=['custom.css']

html_sidebars = {
    "**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
    "software_reference/ingredients/ingredients": [],
}
