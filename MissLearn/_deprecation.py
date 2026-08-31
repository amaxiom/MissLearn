# -*- coding: utf-8 -*-
"""Deprecation machinery, so the policy in CONTRIBUTING.md is enforceable.

The policy is scikit-learn's: a ``FutureWarning`` naming the replacement,
kept for two minor releases before removal. A policy that exists only in prose
tends to be applied inconsistently, and inconsistency is what this project has
been removing everywhere else, so it is given a single implementation here.

``FutureWarning`` rather than ``DeprecationWarning`` is deliberate. Python
hides ``DeprecationWarning`` from end users by default, so it reaches the
library's own developers and nobody else, which is the wrong audience for a
notice about code someone else has written.
"""
import functools
import inspect
import warnings

__all__ = ["deprecated", "deprecate_parameter"]


def deprecated(replacement=None, removed_in=None, extra=""):
    """Mark a function or method as deprecated.

    Parameters
    ----------
    replacement : str, optional
        What to use instead. Name it: a warning that does not say what to do
        next merely tells the user they have a problem.
    removed_in : str, optional
        The version that will drop it, so a reader can plan.
    extra : str, optional
        Any further context.
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = "%s is deprecated" % func.__qualname__
            if removed_in:
                msg += " and will be removed in %s" % removed_in
            if replacement:
                msg += "; use %s instead" % replacement
            msg += "."
            if extra:
                # Callers write *extra* as a sentence, and about half of them
                # end it with a full stop; appending one unconditionally gave
                # "See the guide..".
                tail = extra.strip()
                msg += " " + (tail if tail.endswith((".", "!", "?"))
                              else tail + ".")
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)
        note = "\n\n.. deprecated::\n   %s%s\n" % (
            ("Use %s instead. " % replacement) if replacement else "",
            ("Removed in %s." % removed_in) if removed_in else "")
        wrapper.__doc__ = (func.__doc__ or "") + note
        return wrapper
    return decorate


def deprecate_parameter(name, replacement=None, removed_in=None):
    """Warn when a deprecated keyword is passed, and only when it is passed.

    Warning unconditionally is the usual mistake: it fires for every caller
    including those who never used the parameter, and users learn to filter
    the library's warnings entirely.
    """
    def decorate(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Checking kwargs alone misses the caller who passed it
            # positionally, which is the caller most likely to be running old
            # code: f(x, 5) filled the deprecated slot and heard nothing.
            # Binding against the signature sees both spellings.
            supplied = name in kwargs
            if not supplied:
                try:
                    supplied = name in signature.bind_partial(
                        *args, **kwargs).arguments
                except TypeError:
                    # A call that does not match the signature at all; let
                    # the function raise the real error rather than warn.
                    supplied = False
            if supplied:
                msg = "The '%s' parameter of %s is deprecated" % (
                    name, func.__qualname__)
                if removed_in:
                    msg += " and will be removed in %s" % removed_in
                if replacement:
                    msg += "; use '%s' instead" % replacement
                warnings.warn(msg + ".", FutureWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorate
