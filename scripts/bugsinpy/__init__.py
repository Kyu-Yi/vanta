"""BugsInPy harness for Vanta — ground-truth-labelled repair sessions.

Design: ``plans/bugsinpy.md``. This package is the harness; the pipeline itself is
untouched except where the design says so.

[VANTA-WHY] Every module here obeys one rule that shapes the whole layout: nothing
third-party runs on the host. BugsInPy source is untrusted, and the tests Vanta's
Attacker writes are untrusted too. So each checkout, each native suite run and each
oracle run goes through ``shuru`` (this package's ``shuru`` module), and the only
things that execute on the host are ``git``, ``shuru`` and Vanta itself.
"""

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
