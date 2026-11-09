"""Vanta's pytest exit plugin: end the process the moment pytest is done.

[VANTA-WHY] M3 §8.6 round 13, item 1. A project whose own tests start a
non-daemon worker thread and never stop it (luigi's ``test/worker_test.py``
starts ``Worker.run`` in a ``threading.Thread`` and, unlike the older trees,
never calls ``Worker.stop``) leaves the interpreter alive *after* pytest has
finished: pytest writes its JSON report and prints its summary, then ``python``
blocks in orderly interpreter shutdown waiting for that thread. The harness
measures a run by the markers its own shell script prints *after* the pytest
command returns (``__VANTA_EXIT_*__`` / ``__VANTA_T1_*__``), so a lingering
thread means those markers are never printed, the host-side wall-clock cap
fires, and a suite whose report is complete and fully readable is recorded as
``timed out (hit the cap)`` with ``run_secs=null``.

By the time ``pytest`` is finished the report is already on disk —
``pytest_jsonreport`` writes it in ``pytest_sessionfinish`` (``tryfirst``), and
its ``pytest_unconfigure`` merely unregisters itself — so the honest fix is to
end the process in ``pytest_unconfigure`` with the status pytest itself would
have exited with. Nothing the harness measures moves: the same command runs,
the same report is written, the same exit status is reported; only interpreter
shutdown, which nothing measures and which the suite does not control, is taken
out of the picture.

[VANTA-WHY] The file is written into the VM at run time and never into the
graded tree or the layer, so it cannot reach the model's context, the tree
fingerprints or the cache keys. It travels two ways off ONE source of bytes:
``scripts/bugsinpy/suite.py`` reads this file and writes it into the guest with
a heredoc, and ``src/pipeline/harness.rs`` compiles the same bytes in with
``include_str!``. A second *copy* of the rule would drift; a second *spelling*
of one source cannot.
"""

import os
import sys

import pytest

_exit_status = None


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Remember the status pytest is about to exit with.

    Deferred to ``pytest_unconfigure`` rather than exiting here so that every
    other ``pytest_sessionfinish`` hook — above all ``pytest_jsonreport``'s,
    which writes the JSON report the harness reads back — has already run.
    """
    global _exit_status
    _exit_status = int(session.exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """End the process, after the report is on disk and every other hook ran."""
    if _exit_status is None:
        # pytest never reached ``pytest_sessionfinish`` (an internal error, or a
        # ``pytest_unconfigure`` that ran without a session). Leaving the
        # interpreter's own exit path alone is the neutral choice: this plugin
        # reports pytest's status, and it never learned one.
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(_exit_status)
