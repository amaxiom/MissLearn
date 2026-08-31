# -*- coding: utf-8 -*-
"""Run any MissLearn example script, see the output, and keep a copy of it.

The example scripts are ordinary programs: they print, they draw, and they
end. Running one from a notebook usually means either losing the figures to a
separate window or losing the console text once the cell scrolls away. This
module runs a script in-process so its figures appear inline, while writing
the same console text and the same figures to a directory so the run can be
looked at again later.

Used by Example_Explorer.ipynb, which is only a thin widget layer over
:func:`run_example`. It works just as well from a plain prompt::

    import example_explorer as ex
    ex.list_examples()
    ex.run_example("09_secom_blockwise", quick=True)

Notes on what it has to work around
-----------------------------------
* Five of the ten scripts parse ``--quick`` with argparse. In a notebook
  ``sys.argv`` belongs to the kernel and argparse would choke on it, so argv
  is replaced for the duration of the run.
* Every script ends in ``if __name__ == "__main__": main()``, except
  ``05_credit_approval_benchmark``, whose work happens at module level.
  Running under ``run_name="__main__"`` covers both.
* ``plt.show()`` is replaced while a script runs so each figure can be
  displayed and written to disk before it is closed. Left alone, an inline
  backend closes figures on show and there is nothing left to save.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "explorer_output"

# Scripts that are examples. real_data_fair_benchmark.py is a benchmark
# harness rather than a worked example, and the gallery excludes it too.
_EXCLUDE = {"example_explorer.py", "real_data_fair_benchmark.py",
            "__init__.py"}


def _title_of(path: Path) -> str:
    """First line of the module docstring, which is the example's title."""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, UnicodeDecodeError):
        return ""
    return (doc or "").strip().split("\n")[0].strip()


def _supports_quick(path: Path) -> bool:
    try:
        return "--quick" in path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def discover():
    """Every runnable example, in filename order.

    Returns a list of dicts with ``name``, ``path``, ``title`` and ``quick``.
    """
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.name in _EXCLUDE or p.name.startswith("_"):
            continue
        if not p.name[0].isdigit():
            continue
        out.append({"name": p.stem, "path": p, "title": _title_of(p),
                    "quick": _supports_quick(p)})
    return out


def list_examples():
    """Print the available examples, whether each accepts ``--quick``.

    Titles are printed whole. They run to about 75 characters and were
    previously cut at 60, which removed the end of most of them, and the end
    is usually the part that says what the example is actually for.
    """
    rows = discover()
    name_w = max([len(r["name"]) for r in rows] + [len("example")])
    title_w = max([len(r["title"]) for r in rows] + [len("what it shows")])
    print("%-*s  %-5s  %s" % (name_w, "example", "quick", "what it shows"))
    print("-" * (name_w + 2 + 5 + 2 + title_w))
    for r in rows:
        print("%-*s  %-5s  %s"
              % (name_w, r["name"], "yes" if r["quick"] else "-", r["title"]))
    print()
    print("run_example(\"<example>\", quick=True) to run one.")
    return rows


def picker_labels(rows=None):
    """``(label, name)`` pairs for a dropdown: the name only.

    Titles are deliberately not included. A dropdown clips its text to
    whatever width the browser gives the control, and no layout hint
    reliably prevents that, so a label carrying the title lost the end of it
    however wide the widget was asked to be. The title is not missing from
    the notebook: the table printed by :func:`list_examples` above the picker
    shows every one in full, and the picker puts the selected title
    underneath itself where nothing truncates it.
    """
    rows = rows if rows is not None else discover()
    return [(r["name"], r["name"]) for r in rows]


class _Tee(io.TextIOBase):
    """Write to several streams at once.

    The notebook's own stdout has to keep receiving text so the run can be
    watched as it happens; the buffer is what ends up in console.txt.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except ValueError:      # a closed notebook stream
                pass


class Result:
    """What a run produced."""

    def __init__(self, name, ok, seconds, out_dir, n_figures, console,
                 error=None):
        self.name = name
        self.ok = ok
        self.seconds = seconds
        self.out_dir = out_dir
        self.n_figures = n_figures
        self.console = console
        self.error = error

    def __repr__(self):
        state = "ok" if self.ok else "FAILED: %s" % self.error
        where = (" -> %s" % self.out_dir) if self.out_dir else ""
        return ("<%s  %s  %.1fs  %d figure(s)%s>"
                % (self.name, state, self.seconds, self.n_figures, where))


def _resolve(example):
    rows = {r["name"]: r for r in discover()}
    if example in rows:
        return rows[example]
    # Accept "09", "09_secom_blockwise.py", or a unique fragment.
    stem = str(example).removesuffix(".py")
    if stem in rows:
        return rows[stem]
    hits = [r for k, r in rows.items() if stem and stem in k]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        "no example matches %r. Available: %s"
        % (example, ", ".join(sorted(rows))))


def run_example(example, quick=False, save=True, show=True, out_root=None,
                timestamp=None):
    """Run one example script.

    Parameters
    ----------
    example : str
        Name, filename, or a unique fragment such as ``"09"``.
    quick : bool
        Pass ``--quick`` where the script accepts it. Ignored, with a note,
        where it does not: five of the ten have no such option and will run
        at full size.
    save : bool
        Write the console text and every figure to ``explorer_output/``.
    show : bool
        Display figures inline as they are produced.
    out_root : path-like, optional
        Where to write. Defaults to ``examples/explorer_output``.
    timestamp : str, optional
        Folder name under the example's directory. Defaults to the current
        local time, so repeated runs do not overwrite each other.

    Returns
    -------
    Result
    """
    row = _resolve(example)
    path = row["path"]

    if quick and not row["quick"]:
        print("note: %s has no --quick option; running at full size.\n"
              % row["name"])
        quick = False

    out_dir = None
    if save:
        stamp = timestamp or time.strftime("%Y-%m-%d_%H%M%S")
        out_dir = Path(out_root or OUTPUT_ROOT) / row["name"] / stamp
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from IPython.display import display
    except ImportError:                                   # pragma: no cover
        display = None

    buf = io.StringIO()
    counter = {"n": 0}
    real_show = plt.show

    def _ensure(d):
        """Make sure the output directory is there, immediately before use.

        Creating it once at the start and trusting it to persist is not safe
        enough here: a wine-pipeline run found its directory gone by the time
        the first figure was ready, and lost the figures and the console log
        with it. Whatever removed it, re-checking costs nothing next to
        writing a PNG, and the alternative is losing the whole run's output
        at the last step.
        """
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def _capture_show(*_args, **_kwargs):
        """Save, display, then close whatever is currently open."""
        for num in plt.get_fignums():
            fig = plt.figure(num)
            counter["n"] += 1
            if out_dir is not None:
                try:
                    _ensure(out_dir)
                    fig.savefig(out_dir / ("fig%02d.png" % counter["n"]),
                                dpi=150)
                except OSError as exc:
                    print("could not save figure %d: %s"
                          % (counter["n"], exc))
            if show and display is not None:
                display(fig)
            plt.close(fig)

    argv_before = sys.argv
    cwd_before = os.getcwd()
    path_added = False
    started = time.time()
    ok, err = True, None

    try:
        sys.argv = [str(path)] + (["--quick"] if quick else [])
        # Scripts resolve data and caches relative to their own directory,
        # but run from here anyway so any relative path a script uses lands
        # where it would if it had been run from a shell in examples/.
        os.chdir(HERE)
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
            path_added = True
        if str(HERE.parent) not in sys.path:
            sys.path.insert(0, str(HERE.parent))

        plt.show = _capture_show
        # A script runs with its own plotting settings, not the notebook's.
        #
        # rc_context restores every rcParam afterwards, so a script cannot
        # leak its styling back into the notebook. It also forces the
        # constrained layout engine off for the duration, which is not
        # cosmetic: the scripts fall into two camps, and 06 to 10 select
        # constrained layout themselves and never call tight_layout, while
        # 01 to 04 call tight_layout and expect the default engine. With the
        # notebook's constrained_layout left on, matplotlib refuses to swap
        # engines once a colorbar exists, and 03_wine_pipeline died on
        # exactly that. Scripts in the first camp set the rcParam themselves
        # inside the run, so nothing is taken away from them.
        with mpl.rc_context({"figure.constrained_layout.use": False}), \
                contextlib.redirect_stdout(_Tee(sys.stdout, buf)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, buf)):
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit as exc:                 # argparse --help, etc.
                if exc.code not in (0, None):
                    raise
            # Anything drawn but never shown still counts as a result.
            _capture_show()
    except BaseException as exc:                      # noqa: BLE001
        ok, err = False, "%s: %s" % (type(exc).__name__, exc)
        tb = traceback.format_exc()
        buf.write("\n" + tb)
        print(tb)
    finally:
        plt.show = real_show
        sys.argv = argv_before
        os.chdir(cwd_before)
        if path_added and str(HERE) in sys.path:
            sys.path.remove(str(HERE))

    seconds = time.time() - started
    console = buf.getvalue()

    if out_dir is not None:
        _ensure(out_dir)
        (out_dir / "console.txt").write_text(console, encoding="utf-8")
        (out_dir / "run.txt").write_text(
            "example : %s\nquick   : %s\nstatus  : %s\nseconds : %.1f\n"
            "figures : %d\nerror   : %s\n"
            % (row["name"], quick, "ok" if ok else "failed", seconds,
               counter["n"], err or ""),
            encoding="utf-8")

    res = Result(row["name"], ok, seconds, out_dir, counter["n"], console, err)
    print("\n%s" % res)
    return res
