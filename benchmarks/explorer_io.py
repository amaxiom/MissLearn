# -*- coding: utf-8 -*-
"""Optional saving of explorer output: figures, tables, and a run record.

Both explorer notebooks carried a ``SAVE = True`` that nothing ever read,
directly under a comment claiming nothing was written to disk. This module is
what that switch should have been attached to.

The design goal is that the plotting cells do not change. They call
``plt.show()`` in loops, and rewriting every one of those call sites to be
save-aware would be both tedious and easy to get half-done. Instead
:func:`install` wraps ``plt.show`` once, so every figure a notebook produces
passes through one place that knows whether saving is on and where output is
going. Turning saving off restores ordinary behaviour exactly.

Usage from a notebook::

    import explorer_io as eio
    eio.install()                                  # once, in the setup cell
    eio.begin("MissRidge", save=True)              # at the top of the run
    ...                                            # figures save themselves
    eio.save_table(df, "crossover_regression")     # tables are explicit
    eio.finish()                                   # writes run.txt

Output goes to ``benchmarks/explorer_output/<family>/<timestamp>/``. Runs are
timestamped so a second run of the same family does not overwrite the first.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "explorer_output"

_state = {
    "enabled": False,
    "dir": None,
    "n_fig": 0,
    "n_table": 0,
    "family": None,
    "started": None,
    "saved": set(),
    "preexisting": set(),
    "installed": False,
    "real_show": None,
}


def begin(family, save=True, root=None, timestamp=None, quiet=False):
    """Start a run, creating an output directory when ``save`` is true.

    Calling this again starts a new run: the figure counter resets and a new
    timestamped directory is made, so re-running the notebook from the top
    does not append to the previous run's output.

    Parameters
    ----------
    family : str
        Used as the directory name, so output is grouped by what was run.
    save : bool
        When false nothing is written and figures display as normal. This is
        the switch the notebook checkbox drives.
    root : path-like, optional
        Defaults to ``benchmarks/explorer_output``.
    timestamp : str, optional
        Sub-directory name. Defaults to the current local time.

    Returns
    -------
    pathlib.Path or None
        Where output will go, or None when saving is off.
    """
    _state.update(enabled=bool(save), n_fig=0, n_table=0,
                  family=str(family), started=time.time(), dir=None)
    # A fresh run must not inherit the previous record, and anything already
    # on screen belongs to whatever ran before, not to this run.
    _state["saved"] = set()
    _state["preexisting"] = {id(f) for f in _open_figures()}

    if save:
        stamp = timestamp or time.strftime("%Y-%m-%d_%H%M%S")
        out = Path(root or OUTPUT_ROOT) / str(family) / stamp
        out.mkdir(parents=True, exist_ok=True)
        _state["dir"] = out
        if not quiet:
            print("Saving this run to %s" % out)
    elif not quiet:
        print("Saving is off; nothing will be written to disk.")

    if not _state["installed"]:
        install()
    return _state["dir"]


def _ensure(d):
    """Re-create the output directory immediately before writing to it.

    Creating it once at the start and trusting it to survive is not safe
    enough: an examples run found its directory gone by the time the first
    figure was ready and lost everything it had produced. Re-checking costs
    nothing beside writing a PNG.
    """
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _open_figures():
    """Currently open figures, without disturbing which one is current.

    ``plt.figure(num)`` would activate each figure as a side effect of
    looking at it, which changes what a following ``plt.gca()`` refers to.
    """
    from matplotlib import _pylab_helpers
    return [m.canvas.figure
            for m in _pylab_helpers.Gcf.get_all_fig_managers()]


def _saving_show(*args, **kwargs):
    """Write the figures this run drew, then call the real ``plt.show``.

    Saving happens first because an inline backend closes figures on show,
    which would leave nothing to write.

    Two things are skipped, and both were found by testing under Agg rather
    than reasoned about. A figure already written is skipped, or every later
    show would write the whole open set again. A figure that was already
    open when the run began is skipped, because it belongs to whatever ran
    before and is not this run's output. Under an inline backend neither
    arises, since show closes as it goes; under Agg nothing closes, and
    without both guards two figures across two runs produced ten files.
    """
    if _state["enabled"] and _state["dir"] is not None:
        for fig in _open_figures():
            marker = id(fig)
            if marker in _state["saved"] or marker in _state["preexisting"]:
                continue
            _state["saved"].add(marker)
            _state["n_fig"] += 1
            try:
                _ensure(_state["dir"])
                fig.savefig(_state["dir"] / ("fig%02d.png" % _state["n_fig"]),
                            dpi=150)
            except Exception as exc:                       # noqa: BLE001
                print("could not save figure %d: %s" % (_state["n_fig"], exc))
    return _state["real_show"](*args, **kwargs)


def install():
    """Route ``plt.show`` through the saver. Safe to call more than once."""
    if _state["installed"]:
        return
    _state["real_show"] = plt.show
    plt.show = _saving_show
    _state["installed"] = True


def uninstall():
    """Restore the original ``plt.show``."""
    if _state["installed"]:
        plt.show = _state["real_show"]
        _state["installed"] = False


def save_table(obj, name):
    """Write a DataFrame (or anything with ``to_csv``) beside the figures.

    Tables are explicit rather than intercepted: unlike figures there is no
    single call every one of them passes through, and guessing which
    displayed object was a result worth keeping would be worse than asking.
    """
    if not (_state["enabled"] and _state["dir"] is not None):
        return None
    _ensure(_state["dir"])
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                   for ch in str(name))
    path = _state["dir"] / ("%s.csv" % safe)
    try:
        obj.to_csv(path, index=True)
    except AttributeError:
        path = path.with_suffix(".txt")
        path.write_text(str(obj), encoding="utf-8")
    _state["n_table"] += 1
    return path


def finish(quiet=False):
    """Write the run record and report what was kept."""
    if not (_state["enabled"] and _state["dir"] is not None):
        if not quiet:
            print("Saving was off; nothing written.")
        return None

    seconds = time.time() - (_state["started"] or time.time())
    _ensure(_state["dir"])
    (_state["dir"] / "run.txt").write_text(
        "family  : %s\nseconds : %.1f\nfigures : %d\ntables  : %d\n"
        % (_state["family"], seconds, _state["n_fig"], _state["n_table"]),
        encoding="utf-8")
    if not quiet:
        print("Saved %d figure(s) and %d table(s) to %s"
              % (_state["n_fig"], _state["n_table"], _state["dir"]))
    return _state["dir"]


def output_dir():
    """Where the current run is writing, or None."""
    return _state["dir"]
