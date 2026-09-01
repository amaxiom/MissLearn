# -*- coding: utf-8 -*-
"""Rebuild docs/MissLearn_User_Guide.pdf from docs/USER_GUIDE.md.

The PDF used to be produced by hand, which is why it drifted from the
markdown it is supposed to mirror. This script is the whole procedure, so the
next person regenerating it does not have to reconstruct the invocation.

Requires pandoc and xelatex on PATH. The template in ``_static`` exists
because pandoc's stock LaTeX template needs a 2022-or-later kernel and the
MiKTeX here is older; see the comment at the top of that file.

Usage
-----
    python docs/build_pdf.py            rebuild in place
    python docs/build_pdf.py --check    report whether the PDF is stale
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "USER_GUIDE.md")
TARGET = os.path.join(HERE, "MissLearn_User_Guide.pdf")
TEMPLATE = os.path.join(HERE, "_static", "guide_template.tex")

TITLE = "MissLearn User Guide"
AUTHOR = "Amanda S. Barnard"


def build(date_label):
    for tool in ("pandoc", "xelatex"):
        if shutil.which(tool) is None:
            raise SystemExit("%s is not on PATH; cannot build the PDF." % tool)

    cmd = [
        "pandoc", SOURCE, "-o", TARGET,
        "--pdf-engine=xelatex",
        "--template=" + TEMPLATE,
        "--toc", "--toc-depth=2",
        "--no-highlight",
        "-M", "title=" + TITLE,
        "-M", "author=" + AUTHOR,
        "-M", "date=" + date_label,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # pandoc reports missing glyphs as warnings; they are the one thing that
    # silently degrades a PDF, so they are promoted here rather than buried.
    missing = [L for L in (res.stderr or "").splitlines()
               if "Missing character" in L]
    if res.returncode != 0:
        sys.stdout.write(res.stderr)
        raise SystemExit("pandoc failed with exit %d" % res.returncode)
    if missing:
        for L in missing:
            print("  WARNING %s" % L.strip())
        raise SystemExit(
            "%d character(s) would be dropped from the PDF; add a "
            "\\newunicodechar mapping for each in %s"
            % (len(missing), os.path.relpath(TEMPLATE, HERE)))
    print("wrote %s" % os.path.relpath(TARGET, os.path.dirname(HERE)))


def stale():
    if not os.path.exists(TARGET):
        return True
    return os.path.getmtime(SOURCE) > os.path.getmtime(TARGET)


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("stale" if stale() else "up to date")
        raise SystemExit(1 if stale() else 0)
    # Passed in rather than read from the clock so a rebuild of unchanged
    # source produces an unchanged document.
    label = "August 2026"
    for a in sys.argv[1:]:
        if a.startswith("--date="):
            label = a.split("=", 1)[1]
    build(label)
