# -*- coding: utf-8 -*-
"""Track fit and predict timings, and fail when one regresses.

Why
---
A benchmark harness already exists, but nothing watches the numbers over time,
so a change that quietly makes something slower or worse is invisible until
someone notices. This project has already lost two days to exactly that: a
Gaussian-process comparator was misconfigured, the arm diverged, and the drift
sat undetected because no tracked number moved.

What it records
---------------
Both timings and accuracy. Timing alone would miss a change that keeps the
speed and breaks the answer; accuracy alone would miss the fix that makes a
model correct and fifty times slower. Both matter and both are cheap to store.

Usage
-----
    python track_performance.py --update      # write or refresh the baseline
    python track_performance.py               # compare against it, exit 1 on
                                              # a regression

Timings vary between machines by more than any threshold worth setting, so a
baseline is per-machine and CI compares against one recorded on its own runner.
The accuracy figures are deterministic and portable, and are compared strictly.
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import MissLearn as ML                                        # noqa: E402

BASELINE = os.path.join(_HERE, "performance_baseline.json")

# A slowdown must clear BOTH thresholds to count. A relative threshold alone
# produces nonsense on the fast cases: MissBayesClassifier fits in 12 ms, so
# an extra 12 ms of scheduler noise reads as "+94% regression" and the report
# becomes something people learn to ignore. An absolute floor of 50 ms is
# below anything a user would notice and above the noise on a shared runner.
TIME_TOLERANCE = 0.30          # relative
TIME_FLOOR_S = 0.05            # absolute; both must be exceeded

# Accuracy is compared strictly and needs no floor. It is deterministic, so
# any movement is a real change in the model rather than in the machine.
SCORE_TOLERANCE = 0.005

CASES = [
    ("MissLinear",             dict(compute_se=False),   "regression", 400, 8),
    ("MissLinear_se",          dict(compute_se=True),    "regression", 300, 6),
    ("MissRidgeRegressor",     dict(compute_se=False),   "regression", 400, 8),
    ("MissLASSORegressor",     dict(),                   "regression", 300, 6),
    ("MissBayesRegressor",     dict(),                   "regression", 500, 10),
    ("MissNeighborsRegressor", dict(),                   "regression", 400, 8),
    ("MissLogistic",           dict(compute_se=False),   "classification", 400, 8),
    ("MissBayesClassifier",    dict(),                   "classification", 500, 10),
    ("MissSupportClassifier",  dict(),                   "classification", 300, 6),
]

_CLS = {"MissLinear_se": "MissLinear"}


def dataset(n, p, task, rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(scale=0.4, size=n)
    if task == "classification":
        y = (y > np.median(y)).astype(float)
    Xm = X.copy()
    # MAR, and blockwise in part, so the pattern count stays realistic: cost
    # is driven by the number of distinct patterns, not by the rate.
    for j in range(int(0.6 * p)):
        ref = X[:, (j + 1) % p]
        pr = np.where(ref > np.median(ref), rate * 1.4, rate * 0.6)
        Xm[rng.random(n) < pr, j] = np.nan
    return Xm, y


REPEATS = 3


def measure():
    """Best of REPEATS, after a discarded warm-up.

    A single timing is not a measurement. The first run of this script
    reported MissLinear 41% slower than the run before it with no code change
    at all: import cost, cache state and whatever else the machine was doing.
    The minimum over repeats is the standard estimator here, because timing
    noise is one-sided. Nothing makes a fit spuriously fast, so the smallest
    observation is the closest to the true cost.
    """
    out = {}
    for label, kw, task, n, p in CASES:
        cls = getattr(ML, _CLS.get(label, label))
        X, y = dataset(n, p, task)

        cls(**kw).fit(X, y)                  # warm-up, discarded

        fits, preds = [], []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            est = cls(**kw).fit(X, y)
            fits.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            est.predict(X)
            preds.append(time.perf_counter() - t0)
        t_fit, t_pred = min(fits), min(preds)
        out[label] = {
            "fit_s": round(t_fit, 4),
            "predict_s": round(t_pred, 4),
            "score": round(float(est.score(X, y)), 6),
        }
        print("  %-24s fit %7.3fs  predict %6.3fs  score %.6f"
              % (label, t_fit, t_pred, out[label]["score"]), flush=True)
    return out


def environment():
    import sklearn
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "MissLearn": ML.__version__,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true",
                    help="write the current run as the new baseline")
    args = ap.parse_args()

    print("measuring ...")
    current = measure()

    if args.update or not os.path.exists(BASELINE):
        with open(BASELINE, "w") as fh:
            json.dump({"environment": environment(), "cases": current},
                      fh, indent=2, sort_keys=True)
        print()
        print("baseline written to %s" % os.path.basename(BASELINE))
        if not args.update:
            print("(no baseline existed; nothing to compare against yet)")
        return 0

    with open(BASELINE) as fh:
        base = json.load(fh)

    env_now, env_then = environment(), base.get("environment", {})
    if env_now != env_then:
        print()
        print("note: the environment differs from the baseline, so timing")
        print("comparisons are advisory. Accuracy is still compared strictly.")
        for k in sorted(set(env_now) | set(env_then)):
            if env_now.get(k) != env_then.get(k):
                print("   %-10s %s -> %s" % (k, env_then.get(k), env_now.get(k)))

    # Machine load moves every case together. Comparing each case against the
    # median shift separates "this code got slower" from "this machine is
    # busy", which a raw comparison cannot do: a run taken while the rest of
    # the suite was executing showed every case 23% to 98% slower with no code
    # change at all, and every one of those would have been reported.
    shifts = []
    for label, now in current.items():
        was = base["cases"].get(label)
        if was:
            shifts.append((now["fit_s"] - was["fit_s"]) / max(was["fit_s"], 1e-9))
    baseline_shift = float(np.median(shifts)) if shifts else 0.0

    regressions = []
    print()
    if abs(baseline_shift) > 0.10:
        print("  machine is running %+.0f%% against the baseline overall;"
              % (baseline_shift * 100))
        print("  each case is judged against that shift, not against zero.")
        print()
    print("  %-24s %9s %9s %8s %8s"
          % ("case", "base", "now", "change", "vs median"))
    print("-" * 70)
    for label, now in current.items():
        was = base["cases"].get(label)
        if was is None:
            print("  %-24s new case, not in the baseline" % label)
            continue
        d_abs = now["fit_s"] - was["fit_s"]
        dt = d_abs / max(was["fit_s"], 1e-9)
        excess = dt - baseline_shift          # this case, beyond machine load
        ds = now["score"] - was["score"]
        flag = ""
        if excess > TIME_TOLERANCE and d_abs > TIME_FLOOR_S:
            flag += "  SLOWER"
            regressions.append(
                "%s: fit %.3fs -> %.3fs (%+.0f%% overall, %+.0f%% beyond the "
                "machine-wide shift)"
                % (label, was["fit_s"], now["fit_s"], dt * 100, excess * 100))
        if abs(ds) > SCORE_TOLERANCE:
            flag += "  SCORE MOVED"
            regressions.append("%s: score %.6f -> %.6f (%+.6f)"
                               % (label, was["score"], now["score"], ds))
        print("  %-24s %8.3fs %8.3fs %+7.0f%% %+7.0f%%%s"
              % (label, was["fit_s"], now["fit_s"], dt * 100,
                 excess * 100, flag))

    print()
    if regressions:
        print("%d regression(s):" % len(regressions))
        for r in regressions:
            print("   " + r)
        print()
        print("A moved score is the serious one. It means the model changed,")
        print("not that the machine was busy.")
        return 1
    print("no regressions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
