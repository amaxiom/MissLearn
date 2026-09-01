## What this changes

## Why

## Checklist

- [ ] `pytest tests/unit_test_suite.py -q` passes
- [ ] `pytest tests/conformance_test_suite.py -q -ra` passes
- [ ] If a `KNOWN_FAILURES` entry now reports `XPASS`, it is deleted here
- [ ] A new estimator is registered in the conformance suite
- [ ] Shared behaviour went into `_conformance.py`, not into one class
- [ ] Numbers quoted in documentation come from a run, not from memory
- [ ] No em dashes or en dashes in prose
