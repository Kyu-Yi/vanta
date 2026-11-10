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
have exited with.

[VANTA-WHY] M3 §8.6 round 13 follow-up — correcting the first version. The
plugin originally ended the process with a bare ``os._exit`` and called itself
"measurement-neutral by construction". That was wrong, and round 13 measured the
cost: ``os._exit`` skips ``atexit``, so ``multiprocessing.util._exit_function``
never runs and every process the suite started is left behind — luigi's
``multiprocessing.Manager()`` server (started by ``luigi/mock.py`` in *every*
luigi tree), plus any forked ``TaskProcess``. A second, independent fact was
missed too: pytest's fd-capture keeps a non-``CLOEXEC`` ``os.dup`` of the run's
real stdout/stderr, so a **``fork``ed** descendant inherits that handle. A
surviving ``Manager`` server therefore holds the run's stdout/stderr pipe open,
and both host-side readers (``scripts/bugsinpy/shuru.py`` and
``src/sandbox/shuru.rs``) read that pipe to EOF — so the host waits out the
wall-clock cap even though pytest wrote a complete, identical report and exited.
What looked like "only interpreter shutdown moved" was in reality "the children
that held the measured pipe stopped being reaped": round 12 was clean only
because normal shutdown ran ``atexit`` and killed them first.

The fix makes the cleanup explicit and independent of ``atexit``. At import the
plugin puts itself in **its own process group** (so the group is exactly this
run's process subtree — the wrapping shell the harness times is *not* in it and
is never signalled). Just before ``os._exit`` it forks a detached reaper that
signals that whole group — ``SIGTERM``, a short grace, then ``SIGKILL`` — which
reaches every process the run started, because children inherit the group unless
they deliberately ``setsid``. The reaper calls ``setsid`` (leaving the group it
is about to signal, so it cannot kill itself) and points its stdio at
``/dev/null`` (so it neither holds the run's pipe nor delays the shell); the
plugin's own process leaves through ``os._exit`` and is protected from the group
``SIGTERM`` by ``SIG_IGN`` for the few microseconds that remain. When
``setpgrp`` is unavailable the plugin says so on stderr with a
``__VANTA_PGRP_FALLBACK__`` marker and falls back to the documented, best-effort
``multiprocessing.active_children()`` reaping.

[VANTA-WHY] The file is written into the VM at run time and never into the
graded tree or the layer, so it cannot reach the model's context, the tree
fingerprints or the cache keys. It travels two ways off ONE source of bytes:
``scripts/bugsinpy/suite.py`` reads this file and writes it into the guest with
a heredoc, and ``src/pipeline/harness.rs`` compiles the same bytes in with
``include_str!``. A second *copy* of the rule would drift; a second *spelling*
of one source cannot.
"""

import os
import signal
import sys
import time

import pytest

_exit_status = None

#: The process group this interpreter made for itself, or ``None`` when
#: ``os.setpgrp`` was unavailable. ``None`` turns on the loud fallback below.
_GROUP = None
_PGRP_ERROR = None
try:
    os.setpgrp()
    _GROUP = os.getpid()
except OSError as exc:
    _PGRP_ERROR = exc

#: How long a child gets to react to ``SIGTERM`` before the reaper escalates.
#: The reaper is detached, so this grace never enters the wall clock the harness
#: measures; it only bounds how long a stubborn child may hold the run's pipe.
_GRACE_SECS = 0.5


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
    except Exception:
        pass
    # The group signal we are about to send would otherwise reach this process
    # too; ``os._exit`` follows within microseconds, so ignoring ``SIGTERM`` here
    # cannot outlive the run and cannot change the status the shell records.
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (OSError, ValueError):
        pass
    if _GROUP is not None:
        _fork_reaper(_kill_group, _GROUP)
    else:
        _fallback()
    os._exit(_exit_status)


def _fork_reaper(work, *args):
    """Fork a detached child that runs ``work(*args)`` and then exits.

    The parent returns at once — its caller then ``os._exit``s, so the shell the
    harness times is never delayed by the reaper's grace period. The child
    ``setsid``s (leaving the group it is about to signal, so it cannot kill
    itself) and sends its stdio to ``/dev/null`` (it must not hold the run's
    stdout/stderr pipe, the very handle whose EOF the host is waiting for).
    """
    try:
        pid = os.fork()
    except OSError:
        return
    if pid != 0:
        return
    try:
        os.setsid()
    except OSError:
        pass
    _detach_stdio()
    try:
        work(*args)
    finally:
        os._exit(0)


def _kill_group(pgid):
    """``SIGTERM`` then, after a short grace, ``SIGKILL`` every member of *pgid*."""
    try:
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(_GRACE_SECS)
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _kill_pids(pids):
    """The fallback's per-pid twin of :func:`_kill_group`."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(_GRACE_SECS)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _detach_stdio():
    """Point fds 0, 1 and 2 at ``/dev/null`` so the reaper holds no run pipe."""
    try:
        null = os.open(os.devnull, os.O_RDWR)
    except OSError:
        return
    for fd in (0, 1, 2):
        try:
            os.dup2(null, fd)
        except OSError:
            pass
    if null > 2:
        try:
            os.close(null)
        except OSError:
            pass


def _fallback():
    """Best-effort cleanup when ``setpgrp`` was unavailable — and never silent.

    The ``__VANTA_PGRP_FALLBACK__`` line on stderr is deliberate: it makes a run
    that could not take the group path show up in the harness's own record, so
    the fallback's frequency is observable rather than guessed. This path cannot
    see a grandchild that is not a tracked ``multiprocessing`` child; it is the
    documented degradation, not the design.
    """
    print("__VANTA_PGRP_FALLBACK__ setpgrp failed:", _PGRP_ERROR, file=sys.stderr)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    try:
        import multiprocessing
    except Exception:
        return
    try:
        pids = [child.pid for child in multiprocessing.active_children() if child.pid]
    except Exception:
        return
    if pids:
        _fork_reaper(_kill_pids, pids)
