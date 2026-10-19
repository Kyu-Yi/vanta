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
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


class ShuruError(RuntimeError):
    """The sandbox itself failed to start or to report a result."""


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


def checkpoint_create(
    name: str,
    script: str,
    *,
    base: Optional[str] = None,
    disk_mb: int = 8192,
    allow_net: bool = True,
    timeout_secs: int = 7200,
    marker: Optional[str] = None,
) -> ExecResult:
    """Bake a checkpoint, then verify it by booting it.

    ``marker`` is a path inside the VM that the script must have created (the
    existing Vanta pattern is ``/opt/vanta-deps-ok``). Without it a failed
    install would be cached under a valid-looking name and every later run would
    report ``InfraFailure`` for a reason that has nothing to do with the model.
    """
    args = [_shuru_bin(), "checkpoint", "create", name]
    if base is not None:
        args += ["--from", base]
    args += ["--disk-size", str(disk_mb)]
    if allow_net:
        args += ["--allow-net"]
    args += ["--", "sh", "-c", script]

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
        [f"cd {shlex.quote(guest_cwd)}"]
        + ([f"export {k}={shlex.quote(v)}" for k, v in env.items()] if env else [])
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
