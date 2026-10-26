"""Thin wrapper over the ``shuru`` CLI.

[VANTA-WHY] The flags here mirror ``src/sandbox/shuru.rs`` on purpose. Vanta's own
sandbox backend is the reference implementation for "how Vanta runs untrusted code
on this machine", and a harness that invented its own invocation would drift from
it — different disk sizes, different mount shape — and then disagree with the
pipeline it is supposed to be measuring. Two facts from that file matter enough to
repeat:

* ``shuru run --mount HOST:GUEST`` requires ``HOST`` to be **inside the process
  CWD**. That is why every tree and every batch directory lives under
  ``runs/bugsinpy/`` inside the Vanta repo rather than in ``/tmp``.
* A checkpoint build is the **only** step that needs ``--allow-net``. Once a
  checkpoint exists, ``shuru run --from <cp>`` boots with no network device at
  all, which is what makes the per-run sandbox genuinely offline.

``shuru checkpoint create`` saves the checkpoint **even when the inner command
fails** (observed upstream, documented in ``shuru.rs::verify_deps_marker``). Every
bake in this harness therefore ends with a sentinel ``touch`` and is verified by
booting the result, so a broken environment cannot be cached and silently poison a
whole wave.

One thing is deliberately **not** mirrored: the VM's size. ``shuru.rs`` passes no
``--cpus``/``--memory`` and so boots whatever shuru defaults to (2 CPUs, 2048 MB), which
is enough to run a suite and not enough to build one; a bake asks for
:data:`DEFAULT_CPUS`/:data:`DEFAULT_MEMORY_MB` instead.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from . import redact


class ShuruError(RuntimeError):
    """The sandbox itself failed to start or to report a result."""


DEV_SHM_PATH = "/dev/shm"
"""Where CPython's POSIX semaphores live, and where the VM puts nothing.

[VANTA-WHY] A Shuru microVM boots with no ``/dev/shm`` at all. CPython's
``multiprocessing`` allocates its semaphores and its shared-memory segments as files
*in that directory* (``multiprocessing/synchronize.py`` -> ``_multiprocessing.SemLock``
-> ``sem_open``), so on such a VM the very first ``multiprocessing.Lock()`` raises
``FileNotFoundError: [Errno 2] No such file or directory`` **with no filename** — the
errno comes from the kernel, not from a failed open of a named path.

Measured, and the shape is why it cost a debugging session: it is the same error text a
missing compiler produces, so it read as a toolchain failure. ``DISTUTILS_DEBUG=1`` on
numpy 1.18.5 showed the traceback ending in ``numpy/distutils/ccompiler.py`` ->
``multiprocessing.pool.ThreadPool`` -> ``SemLock``; the archiver probe had already
cleared ``ar``/``ARFLAGS``. The probe that settled it: ``ls /dev/shm`` -> *No such file
or directory*, ``multiprocessing.Lock()`` -> the same ``FileNotFoundError``, and after
``mkdir -p /dev/shm && mount -t tmpfs ... /dev/shm`` the lock is created.

The blast radius is the whole benchmark, not the bake: any ``Pool``, ``Queue`` or
``Lock`` inside a VM fails this way, so a bug whose tests use multiprocessing reports
*false failures* that look like model errors. It is therefore fixed at every boot rather
than around numpy.
"""

DEV_SHM_ENV = "VANTA_DEV_SHM"
"""The variable name :data:`DEV_SHM_GUARD` exports, and the one the bake records.

Named once because three readers depend on the exact spelling: the guard that sets
it, :func:`bugsinpy.bake.build_env_lines`' filter and assertion, and anyone reading
``/opt/vanta-build-env.txt`` afterwards. A second literal in the bake would not fail
loudly — it would silently stop matching the guard and drop the fact from the record.
"""

DEV_SHM_MOUNTED = "tmpfs mounted"
"""The value :data:`DEV_SHM_GUARD` exports as :data:`DEV_SHM_ENV` once the mount is in place.

A value rather than a bare flag so a recipe can record it with the same ``NAME=value``
shape as everything else in the build-environment record, which is the only shape that
file's reader keeps (:func:`bugsinpy.bake.layer_provenance_lines`).
"""

DEV_SHM_MARKER = f"vanta dev/shm: {DEV_SHM_MOUNTED}"
"""The line every VM prints on **stderr** after its start-up check passes.

Stderr and not stdout: ``shuru.run``'s stdout is pytest's, and the harness parses it
(``_exit_code``, ``parse_collected``). A start-up banner on stdout would be read as
test output.
"""

STARTUP_CHECK_EXIT = 86
"""Exit code of a VM whose start-up check could not run: no ``/dev/shm``.

Distinct from 1 because it is not the command's failure: ``checkpoint_create``'s marker
probe distinguishes it so a VM that cannot mount a tmpfs does not read as "the marker is
missing" and delete a layer that is fine.
"""

MP_SMOKE_EXIT = 87
"""Exit code of a VM where ``/dev/shm`` mounted but multiprocessing still cannot use it."""

MP_SMOKE_PREFIX = "multiprocessing smoke ok"
"""What the smoke check prints when ``Lock()`` and a 2-worker ``Pool`` both work."""

MP_SMOKE_CODE = (
    "import multiprocessing as mp; "
    # Lock() is the direct translation of the failure: it is the call that raises
    # FileNotFoundError when the semaphore directory is absent.
    "lock = mp.Lock(); "
    # A Pool is the indirect one and the reason this is not Lock-only: ThreadPool is
    # what numpy 1.18.5's build_clib reaches through numpy/distutils/ccompiler.py, so
    # a check that only made a Lock would not have reproduced the shipped traceback.
    "pool = mp.Pool(2); "
    "assert pool.map(abs, [-1, -2, -3]) == [1, 2, 3], 'Pool.map'; "
    "pool.close(); pool.join()"
)
"""The one-liner that fails when a VM's ``multiprocessing`` has no semaphore store.

Also run by the bake's toolchain smoke (:func:`bugsinpy.bake.toolchain_smoke_lines`),
which is why it is public: two spellings of this check would drift, and the check is
the whole point of the constant.
"""

DEV_SHM_GUARD = "; ".join(
    [
        # Idempotent: `run` boots a fresh VM from a checkpoint every time, and a
        # checkpoint holds no mounts, but the same guard also runs on the boot that
        # *created* a layer, where a later line could in principle have mounted it.
        f"{DEV_SHM_ENV}=''",
        f"if grep -q '^tmpfs {DEV_SHM_PATH} ' /proc/mounts 2>/dev/null; then "
        f"{DEV_SHM_ENV}='{DEV_SHM_MOUNTED}'",
        f"elif mkdir -p {DEV_SHM_PATH} && mount -t tmpfs -o rw,nosuid,nodev tmpfs "
        f"{DEV_SHM_PATH}; then {DEV_SHM_ENV}='{DEV_SHM_MOUNTED}'",
        "fi",
        f"export {DEV_SHM_ENV}",
        f'[ -n "${DEV_SHM_ENV}" ] || {{ echo "vanta dev/shm: the VM has no '
        f"{DEV_SHM_PATH} and a tmpfs cannot be mounted on it, so CPython's "
        f'multiprocessing cannot create a semaphore" >&2; exit {STARTUP_CHECK_EXIT}; }}',
        f"echo {shlex.quote(DEV_SHM_MARKER)} >&2",
        # The smoke, on whichever interpreter this VM has: `python3` first because the
        # base image's is 3.13 while a layer's `python` is the venv wrapper, and either
        # proves what this check is about (the *kernel* has a semaphore filesystem).
        # Skipped, not failed, when the VM has no interpreter at all: a VM that will
        # never run Python cannot have a Python multiprocessing problem, and the base
        # image before `create_python_checkpoint` installs it is exactly that VM.
        #
        # `if`/`elif` rather than `$(command -v python3 || command -v python || true)`:
        # this guard also runs inside recipe shells that set ``-e``, where an
        # AND-OR list that fails is the list's status and would end the build — and a
        # trailing ``|| true`` would be a second one in a recipe whose own test asserts
        # there is exactly one (:func:`bugsinpy.requirements_recipe`'s).
        'VANTA_MP_PY=""',
        'if command -v python3 >/dev/null 2>&1; then VANTA_MP_PY="python3"',
        'elif command -v python >/dev/null 2>&1; then VANTA_MP_PY="python"',
        "fi",
        f'export VANTA_MP_CODE={shlex.quote(MP_SMOKE_CODE)}',
        '[ -z "$VANTA_MP_PY" ] || "$VANTA_MP_PY" -c "$VANTA_MP_CODE" || '
        f'{{ echo "vanta {MP_SMOKE_PREFIX}: FAILED -- {DEV_SHM_PATH} is mounted but '
        f'multiprocessing cannot use it (Lock and Pool are what every test in this VM '
        f'would fail on)" >&2; exit {MP_SMOKE_EXIT}; }}',
        f'[ -z "$VANTA_MP_PY" ] || echo "vanta {MP_SMOKE_PREFIX}: Lock and a 2-worker '
        'Pool" >&2',
    ]
)
"""Mount a tmpfs on :data:`DEV_SHM_PATH`, prove multiprocessing works, or stop the boot.

[VANTA-WHY] One string, run first by every entry point that boots a VM
(:func:`checkpoint_create`, :func:`run`), because the two facts it establishes are facts
about *the VM* and not about any one command: mounts do not survive into a checkpoint, so
"the layer has /dev/shm" is not a property that could be baked in and read later — it
has to be re-established on every boot, and a boot that cannot establish it must fail
loudly rather than let the command under it report a false test failure.

Fail *before* the command runs, with :data:`STARTUP_CHECK_EXIT`/:data:`MP_SMOKE_EXIT`,
because the alternative is the failure this exists to end: an ``[Errno 2]`` twenty
minutes into a source build, naming no file and no variable.
"""


def startup_check_lines() -> list[str]:
    """The start-up check as recipe lines, for a caller that records rather than boots.

    The bake's interpreter layer wants the same check *inside* its own record
    (``/opt/vanta-build-env.txt``) as well as at boot: the boot says "this VM has
    /dev/shm", the record says "the layer that was baked carries that fact", and those
    are two different claims about two different artifacts.
    """
    return [DEV_SHM_GUARD]


@dataclass
class ExecResult:
    """One command's outcome inside a microVM."""

    stdout: str
    stderr: str
    exit_code: Optional[int]
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def shuru_available() -> bool:
    return shutil.which("shuru") is not None


def _shuru_bin() -> str:
    exe = shutil.which("shuru")
    if exe is None:
        raise ShuruError(
            "the `shuru` CLI is not on PATH; the BugsInPy harness runs every "
            "third-party command in Shuru (plans/bugsinpy.md C3)"
        )
    return exe


def checkpoint_exists(name: str) -> bool:
    out = subprocess.run(
        [_shuru_bin(), "checkpoint", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return False
    return any(line.split() and line.split()[0] == name for line in out.stdout.splitlines())


def checkpoint_delete(name: str) -> None:
    subprocess.run(
        [_shuru_bin(), "checkpoint", "delete", name],
        capture_output=True,
        text=True,
        check=False,
    )


DEFAULT_CPUS = 8
DEFAULT_MEMORY_MB = 8192
"""The VM a *bake* boots with, and this wrapper's one deliberate divergence from
``src/sandbox/shuru.rs``.

[VANTA-WHY] The Rust backend and this wrapper both otherwise let shuru pick its own
defaults, and those are 2 CPUs and 2048 MB — measured in the boot line of every bake
layer's log (``shuru: booting VM (2cpus, 2048MB RAM, 16384MB disk)``). That is enough to
*run* a bug's suite, which is all the backend ever asks of a VM, and it is not enough to
*build* one: measured on ``pandas/1``, ``scipy``'s ``sparse/sparsetools`` C++ translation
units are compiled in parallel by numpy's distutils, the two of them together exceed
2048 MB, and the kernel kills ``cc1plus`` (``g++: fatal error: Killed signal terminated
program cc1plus``) — a build failure that names no source file and reads as a compiler
bug. Sizing reaches the VM through ``checkpoint create``'s own ``--cpus``/``--memory``
and touches no recipe string, so a layer baked under the old defaults stays valid.
"""

BUILD_JOB_ENV = "NPY_NUM_BUILD_JOBS"
"""How many parallel compile jobs numpy's distutils may start.

[VANTA-WHY] CPUs and memory are not the same constraint, and the *parallel* build is
bounded by the smaller of the two. numpy's ``ccompiler._cpu_count`` reads this variable
and otherwise returns ``os.cpu_count()``, so on an 8-CPU VM it starts eight ``cc1plus``
processes — roughly a gigabyte each for scipy's template-heavy C++ — which is more than
8192 MB before the VM's own overhead. The cap is derived from the memory
(:func:`build_jobs`) rather than fixed, because memory is the resource that ran out.
"""

_MB_PER_BUILD_JOB = 2048
"""What one ``cc1plus`` of the largest measured build (scipy's sparsetools) is allowed.

A cap on parallelism, not a measurement of the compiler: the point is only that the
*number* of jobs cannot exceed what the VM's memory holds, and 2 GB per job is the value
that leaves even the 2048 MB default able to run one.
"""


def build_jobs(cpus: int = DEFAULT_CPUS, memory_mb: int = DEFAULT_MEMORY_MB) -> int:
    """How many parallel compile jobs a VM of this size may run: the smaller of the two.

    At the defaults — 8 CPUs, 8192 MB — this is 4, which is what keeps scipy's
    ``sparsetools`` build inside a VM sized for it.
    """
    return max(1, min(cpus, memory_mb // _MB_PER_BUILD_JOB))


VM_SIZING_PREFIX = "vanta bake VM:"
"""What :func:`vm_sizing_guard` prints, which is what makes the sizing a *record*.

The boot prelude's output is the layer's own captured output, so the line reaches
``runs/bugsinpy/bake-logs/<layer>.log`` without any recipe having to know the VM's size.
"""


def vm_sizing_guard(cpus: int = DEFAULT_CPUS, memory_mb: int = DEFAULT_MEMORY_MB) -> str:
    """The boot-prelude half of the sizing: cap the parallel build, and say so.

    [VANTA-WHY] Exported here rather than inside a layer's recipe, because sizing is the
    *VM's* and recipes are the *layers*': a recipe that exported this variable would be a
    different recipe, so every layer baked before the cap existed would have to be
    rebuilt to stay comparable with the ones baked after it. The boot prelude is already
    where a fact about the VM is established (:data:`DEV_SHM_GUARD`) and it runs on the
    same boot as the recipe, in the same shell, so the export reaches every build either
    way.
    """
    jobs = build_jobs(cpus, memory_mb)
    return "; ".join(
        [
            f"export {BUILD_JOB_ENV}={jobs}",
            f"echo '{VM_SIZING_PREFIX} {cpus} cpus, {memory_mb} MB RAM, "
            f"{BUILD_JOB_ENV}={jobs}'",
        ]
    )


def checkpoint_create(
    name: str,
    script: str,
    *,
    base: Optional[str] = None,
    disk_mb: int = 8192,
    allow_net: bool = True,
    timeout_secs: int = 7200,
    marker: Optional[str] = None,
    cpus: int = DEFAULT_CPUS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> ExecResult:
    """Bake a checkpoint, then verify it by booting it.

    ``marker`` is a path inside the VM that the script must have created (the
    existing Vanta pattern is ``/opt/vanta-deps-ok``). Without it a failed
    install would be cached under a valid-looking name and every later run would
    report ``InfraFailure`` for a reason that has nothing to do with the model.

    ``cpus``/``memory_mb`` are shuru's own ``--cpus``/``--memory`` and are the *VM's*
    size rather than the layer's contents (:data:`DEFAULT_CPUS`), so a layer baked
    before they were passed is still the layer this call would produce.
    """
    args = [_shuru_bin(), "checkpoint", "create", name]
    if base is not None:
        args += ["--from", base]
    args += ["--disk-size", str(disk_mb), "--cpus", str(cpus), "--memory", str(memory_mb)]
    if allow_net:
        args += ["--allow-net"]
    # [VANTA-WHY] The start-up check runs *before* the recipe, on the same boot, so a
    # VM that cannot provide /dev/shm fails here rather than inside whatever build
    # the recipe reaches first (DEV_SHM_GUARD). A mount cannot be baked into the
    # layer, so this is not redundant with anything the recipe records. The sizing's
    # half follows it for the same reason and on the same boot: the export has to be in
    # the shell that runs the recipe, and the echo has to reach the layer's log.
    args += [
        "--",
        "sh",
        "-c",
        f"{DEV_SHM_GUARD}\n{vm_sizing_guard(cpus, memory_mb)}\n{script}",
    ]

    created = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_secs,
        stdin=subprocess.DEVNULL,
    )
    result = ExecResult(
        stdout=created.stdout,
        stderr=created.stderr,
        exit_code=created.returncode,
        timed_out=False,
    )
    if not result.ok:
        # Measured: shuru saves the checkpoint *before* reporting that the inner
        # command failed (`shuru: checkpoint 'x' saved` on an exit-127 shell). Left
        # in place, that layer looks baked, so the next bake would skip it and
        # every run would inherit a broken environment.
        checkpoint_delete(name)
        return result

    if marker is not None:
        # [VANTA-WHY] Twice before condemning the layer. A missing marker is
        # deterministic (the install failed), but a *failed boot* is not: the same
        # probe that reports the marker present on one boot can fail on the next,
        # and deleting on that race throws away minutes of installing. Both
        # attempts' output is carried into the error, because the first version of
        # this check reported only "marker missing" while the actual reason (a
        # non-zero `shuru run`) was being discarded — which sent a whole debugging
        # session after the wrong suspect.
        attempts: list[str] = []
        for attempt in (1, 2):
            probe = run(
                checkpoint=name,
                host_root=Path.cwd(),
                command=["sh", "-c", f"test -f {marker}"],
                disk_mb=disk_mb,
                timeout_secs=300,
            )
            if probe.ok:
                return result
            if probe.exit_code == STARTUP_CHECK_EXIT:
                # [VANTA-WHY] The probe boots the layer, so it runs the start-up check
                # too (``run``). A VM whose check fails is a fact about the *machine*,
                # not about the layer the recipe just built: the marker was never
                # looked for. Deleting here would throw away a healthy layer for a
                # tmpfs that could not be mounted on this boot — the failure mode
                # ``checkpoint_create``'s two-attempt loop already guards against for
                # a racy boot, arriving through a different door.
                raise ShuruError(
                    f"checkpoint {name!r} is left in place: the probe VM could not "
                    f"establish {DEV_SHM_PATH} (exit {STARTUP_CHECK_EXIT}), so its "
                    f"marker {marker!r} was never tested. Fix the host's ability to "
                    "boot a VM with a tmpfs (DEV_SHM_GUARD) and re-run.\n"
                    f"--- probe stderr ---\n{probe.stderr.strip()[-1000:]}"
                )
            attempts.append(
                f"attempt {attempt}: exit={probe.exit_code} "
                f"timed_out={probe.timed_out}\n"
                f"--- stdout ---\n{probe.stdout.strip()[-1000:]}\n"
                f"--- stderr ---\n{probe.stderr.strip()[-1000:]}"
            )
        # Neither the build log nor the probe log is dropped: the build log is what
        # shows whether the inner script even reached its `touch`.
        checkpoint_delete(name)
        raise ShuruError(
            f"checkpoint {name!r} built but its marker {marker!r} is missing on two "
            "boots; the checkpoint has been deleted.\n"
            f"--- build log (tail) ---\n{(result.stdout + result.stderr).strip()[-2000:]}\n"
            f"--- probe ---\n" + "\n".join(attempts)
        )
    return result


def run(
    *,
    checkpoint: str,
    host_root: Path,
    command: Sequence[str],
    disk_mb: int,
    guest_cwd: str = "/workspace",
    timeout_secs: int = 900,
    allow_host_writes: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> ExecResult:
    """Run ``command`` inside a fresh microVM started from ``checkpoint``.

    ``host_root`` must be inside the process CWD (Shuru's mount rule). It appears
    inside the VM as ``/workspace``, so the harness keeps its trees under
    ``runs/bugsinpy/`` and addresses them as ``/workspace/<batch>/worktrees/...``.
    """
    host_root = host_root.resolve()
    cwd = Path.cwd().resolve()
    if cwd not in host_root.parents and host_root != cwd:
        raise ShuruError(
            f"shuru requires the mounted host path to be inside the CWD; "
            f"{host_root} is not under {cwd}"
        )

    # [VANTA-WHY] ``command`` is an *argv*, not a shell fragment, so it is quoted
    # (``shlex.join``) rather than joined with ``&&``. The earlier version pasted
    # each element into the shell line verbatim, which is indistinguishable from
    # correct while every element happens to be a bare word — and silently wrong the
    # moment one is not. Measured: the marker probe passes ``["sh", "-c", "test -f
    # /opt/vanta-deps-ok"]``, which the ``&&`` join turned into
    # ``cd /workspace && sh && -c && test -f /opt/vanta-deps-ok``: a bare ``sh``
    # reading the inherited stdin. It blocked until the 300 s timeout on every
    # boot, so the probe reported the marker missing on two healthy checkpoints in
    # a row and the bake deleted a layer that was fine.
    inner = " && ".join(
        # [VANTA-WHY] The start-up check is part of *every* boot, this one included:
        # ``run`` is the path the suite, the runner, the oracle and the marker probe
        # all take, and a mount does not survive into a checkpoint, so "the VM has
        # /dev/shm" is a fact that has to be re-established on each one rather than
        # read back out of the layer (DEV_SHM_GUARD). ``&&`` makes its non-zero exit
        # the command's, so a VM that cannot provide a tmpfs reports the start-up
        # check and not a test failure.
        [DEV_SHM_GUARD, f"cd {shlex.quote(guest_cwd)}"]
        # [VANTA-WHY] The `export` lines are part of the command string, and a
        # command string is printed by `shuru` on failure and visible in `ps`.
        # Redacting here means a secret-named variable — or an `sk-` token inside
        # any value — can never reach either, whatever a future caller passes in
        # `env`. Every dict passed today is a compiler/Python environment, so for
        # them this is byte-for-byte a no-op.
        + ([f"export {k}={shlex.quote(v)}"
            for k, v in redact.redact_env(env).items()] if env else [])
        + [shlex.join(command)]
    )

    args = [
        _shuru_bin(),
        "run",
        "--from",
        checkpoint,
        "--disk-size",
        str(disk_mb),
        "--mount",
        f"{host_root}:/workspace:rw",
    ]
    if allow_host_writes:
        args += ["--allow-host-writes"]
    args += ["--", "sh", "-c", inner]

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_secs,
            # Defence in depth beside the quoting above: a wrong argv can still
            # make the guest's shell read a command, and a *never-EOF* stdin is
            # then an unbounded wait that only the timeout ends. Closed stdin
            # makes such a mistake fail in milliseconds instead.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecResult(
            stdout=_text(exc.stdout),
            stderr=f"shuru run TIMEOUT after {timeout_secs}s",
            exit_code=None,
            timed_out=True,
        )
    return ExecResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        timed_out=False,
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)
