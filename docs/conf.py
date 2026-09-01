# -*- coding: utf-8 -*-
"""Sphinx configuration for the MissLearn documentation site.

Everything documentation lives in this one directory: the guides themselves,
which are the source of truth for prose, and the machinery that renders them
alongside a generated API reference and the example gallery. There was a
separate ``doc/`` holding the machinery for a while, one letter away from this
one, which is a name collision nobody should have to notice.

The guides are markdown and go straight into the toctree; myst renders them in
place. Only CHANGELOG.md and CONTRIBUTING.md need stub pages, because they
live at the repository root and Sphinx will not pull a document into a toctree
from outside its source directory.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

import MissLearn  # noqa: E402

project = "MissLearn"
author = "Amanda S. Barnard"
copyright = "%d, %s" % (date.today().year, author)
version = MissLearn.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
    "myst_parser",          # the guides are markdown, not reStructuredText
]

# The gallery is optional at build time. A contributor editing prose should
# not be blocked by a missing extension, and the API reference and guides are
# complete without it. CI installs doc/requirements.txt, so the published site
# always has it.
try:
    import sphinx_gallery  # noqa: F401
    extensions.append("sphinx_gallery.gen_gallery")
    HAVE_GALLERY = True
except ImportError:
    HAVE_GALLERY = False

autosummary_generate = True
numpydoc_show_class_members = False
autodoc_default_options = {"members": True, "inherited-members": False}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# The ten worked examples are already scripts with a --quick flag; the gallery
# renders them rather than duplicating their content.
if not HAVE_GALLERY:
    # Without the extension the toctree entry would dangle and -W would turn
    # that into a build failure, so the placeholder keeps the tree valid.
    exclude_patterns_extra = ["auto_examples"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"^$",   # build thumbnails without executing by
                                 # default: several examples take 20 minutes
                                 # and CI should not pay that on every commit
    # example_explorer.py is the notebook's runner, not a worked example.
    # Left in, sphinx-gallery lists it in the execution-times table and
    # then cannot cross-reference it, because it has no gallery header.
    "ignore_pattern": r"(real_data_fair_benchmark|example_explorer|__init__)\.py",
    "download_all_examples": False,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}

# The guides carry their own markdown tables of contents linking to headings
# by slug. Without this myst does not create those anchors, so every entry
# becomes a broken cross-reference: 145 warnings on the first build, and with
# -W in CI, a failed build.
myst_heading_anchors = 4

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]
# exclude_patterns_extra is set above when sphinx-gallery is missing.
# It has to be merged in here; assigning exclude_patterns after
# computing it, which is what this file did, silently discarded it and
# left the guard doing nothing.
exclude_patterns += globals().get("exclude_patterns_extra", [])

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "MissLearn %s" % version
html_theme_options = {
    "github_url": "https://github.com/amaxiom/MissLearn",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
