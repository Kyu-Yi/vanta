"""The one scheduler: how many suite VMs one invocation may run at once.

[VANTA-WHY] M3 items S5 and S6. Until now every stage ran its suite runs one
after another on the calling thread, so a machine with 18 cores spent a whole
batch's wall clock inside a single VM at a time. S5 lets ``prepare`` run its
independent suite runs concurrently and S6 lets a bug's independent review rows
overlap, and the mandate is explicit that both — and ``--jobs`` — are governed
by ONE scheduler rather than three.

The unit of the scheduler is a SINGLE SUITE RUN, not a pair and not a bug: the
mandate fixes the price of one run at "``memory_mb`` as allocated plus 1 effective
core per serial suite run", and caps the total at **host cores − 2**. The two
reserved cores are the host's own headroom — the harness, the editor, the shell —
so a batch can never take the machine's last core. The VMs stay at 8 vCPUs
(``shuru`` boots them that way regardless); the accounting here is what changes,
not the layer.

Why a weighted gate and not a thread pool: the runs are not a fixed-size queue.
``prepare`` submits three pairs at once and each pair wants two runs, so the
in-flight count depends on which pairs are mid-pair — and a pool of the wrong
size is either a hidden cap or an overload. A gate that each run acquires for
its own (1 core, memory) pair is the shape the mandate describes and is what
makes ``--prepare-jobs`` mean the same thing at every call site.

The budget is **per invocation**: it is created once by the command and handed
down, so two ``prepare`` processes on one machine do not share it and one batch
cannot starve another. There is deliberately no module-level singleton — a
global budget would make the cap a fact about the process rather than about the
command, and a test that ran two of them would pass on the machine and fail on
CI.

``host_memory_mb`` is best-effort and may be ``None``; the cores cap is the one
the mandate states and the one that always applies. Memory is accounted (it is
half of what a run costs) and the peak is recorded, but an undetectable host
must not make the scheduler refuse to run anything.

M3 item S6 adds a second price, for a ROW rather than a suite run. A row is one
``vanta_m3`` process that boots many microVMs, up to its own peak of simultaneous
ones, so it is charged ``peak_vms × vm_memory_mb`` of memory and one core
(Nathan's "peak × mem, 1 core"). ``peak_vms`` is a conservative knob
(``--peak-vms``, default 12) and the row also records the peak it ACTUALLY
reached, so the charge can be calibrated from real runs instead of assumed. The
charge is accounting only: no VM's configured memory or vCPUs are changed. A row
whose charge alone exceeds the whole memory budget is run ALONE rather than
deadlocked (``_fits`` already had that rule for a single run); the row records
that it was.

``HostSampler`` is the batch-level counterpart Nathan asked for: a row's peak is
attributed per row, but host-wide used memory is not, because the VMs run inside
processes that may not be the row's children. It samples the machine, not a row.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

HOST_CORE_RESERVE = 2
"""The cores this scheduler never allocates — the host keeps them for itself.

[VANTA-WHY] The mandate's cap is "host cores − 2". The reserve is what keeps a
batch from starving the machine it runs on: the harness process, the shell that
launched it, and whatever the operator has open all need a core, and a scheduler
that took all of them would make a concurrent batch *slower* than a serial one
by making every process wait for a core.
"""

DEFAULT_PEAK_VMS = 12
"""M3 item S6 — the conservative per-row peak of simultaneous VMs, the charge knob.

[VANTA-WHY] Nathan's decision: the row price is "(peak × mem, 1 core)" and the
peak is a knob, ``--peak-vms N``, default ``12``. Twelve is the highest peak
observed across 68 recorded ``attempt_*_tests.json`` artifacts (typical 8–12), so
the default over-charges a small row and never under-charges a large one; every
row records the peak it actually reached so the number can be tuned down later.
"""

DEFAULT_VM_MEMORY_MB = 2048
"""M3 item V7 — the memory one VM costs a row that records no ``vm_memory_mb``.

[VANTA-WHY] This is ``shuru.DEFAULT_RUN_MEMORY_MB``: the size a run VM boots at
when a bug's split records no ``vm_memory_mb``. A row that records nothing is not
a row that costs nothing — it boots VMs at this size — so charging ``peak_vms × 0``
would make the least-documented rows the cheapest to run beside each other, which
is the opposite of what a conservative scheduler is for. ``split.validate`` refuses
a recorded size at or below this, so "recorded nothing" and "recorded the default"
are the same VM in practice. The constant is mirrored here the way
``split.VM_MEMORY_MB_MIN`` mirrors it, and a test pins both to
``shuru.DEFAULT_RUN_MEMORY_MB`` so they cannot drift.
"""


def host_cores() -> int:
    """The number of CPUs this host reports, never less than 1."""
    count = os.cpu_count()
    return count if count and count > 0 else 1


def host_cores_budget() -> int:
    """The cores the scheduler may allocate in one invocation."""
    return max(1, host_cores() - HOST_CORE_RESERVE)


def host_memory_mb() -> Optional[int]:
    """Physical memory in MB, or ``None`` when it cannot be read portably.

    Read through ``os.sysconf`` only — the same call resolves the figure on Linux
    and on macOS — and every failure mode (an unsupported name, a non-integer, a
    zero) returns ``None`` rather than a guess. ``None`` means the memory half of
    a run's price is accounted but not capped; the cores cap still holds.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(size, int):
        return None
    if pages <= 0 or size <= 0:
        return None
    return (pages * size) // (1024 * 1024)


def host_load() -> Optional[list[float]]:
    """The host's 1/5/15-minute load average, or ``None`` where it is unavailable."""
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


class Budget:
    """A weighted (cores, memory) gate for one invocation's suite runs.

    ``jobs`` is the invocation's own cap on concurrent runs (``--prepare-jobs``);
    ``cores`` is the smaller of that and the host's own budget, so a ``--jobs``
    larger than the machine is silently limited rather than oversubscribing it.
    ``peak_cores``/``peak_memory_mb`` are the high-water marks, recorded so a row
    can say what the budget actually did and not only what it allowed.
    """

    def __init__(
        self,
        jobs: int,
        *,
        cores_cap: Optional[int] = None,
        memory_cap_mb: Optional[int] = None,
        peak_vms: int = DEFAULT_PEAK_VMS,
    ) -> None:
        self.jobs = max(1, int(jobs))
        self.cap_cores = (
            host_cores_budget() if cores_cap is None else max(1, int(cores_cap))
        )
        self.cap_memory_mb = (
            host_memory_mb() if memory_cap_mb is None else memory_cap_mb
        )
        self.cores = max(1, min(self.jobs, self.cap_cores))
        # M3 item S6 — the per-row charge knob. Stored on the budget so one
        # invocation charges every row the same way, and so a row can state it.
        self.peak_vms = max(1, int(peak_vms))
        self.host_load = host_load()
        self._cond = threading.Condition()
        self._used_cores = 0
        self._used_memory_mb = 0
        self.peak_cores = 0
        self.peak_memory_mb = 0

    @property
    def concurrent(self) -> bool:
        """Whether this invocation may run more than one suite run at a time."""
        return self.cores > 1

    def row_charge_mb(self, vm_memory_mb: Optional[int]) -> int:
        """M3 item S6 — the host memory a ROW is charged: ``peak_vms × vm_memory_mb``.

        A row boots many microVMs at once, so it is charged the whole of the
        simultaneous set it may hold rather than one VM's worth (Nathan: "Charge
        each row its peak simultaneous VMs, not one, or the pool will
        oversubscribe memory"). The multiplication is the charge; no VM's own
        memory is changed.

        [VANTA-WHY] M3 item V7. A row that records no ``vm_memory_mb`` still boots
        VMs, at ``shuru.DEFAULT_RUN_MEMORY_MB``, so it is charged
        ``peak_vms × DEFAULT_VM_MEMORY_MB`` rather than ``peak_vms × 0``. A
        non-positive number is treated as absent for the same reason: the
        accounting must never price a VM at zero, or the rows that say least about
        themselves would be the cheapest to admit beside each other.
        """
        size = int(vm_memory_mb or 0)
        if size <= 0:
            size = DEFAULT_VM_MEMORY_MB
        return self.peak_vms * size

    def charge_exceeds_budget(self, charge_mb: int) -> bool:
        """Whether one row's charge cannot fit the whole memory budget.

        Such a row is run ALONE (``_fits``), which is the closest thing to
        "fits"; the row records the fact so a reader can tell a genuinely serial
        run from one the scheduler could never admit beside another.
        """
        return self.cap_memory_mb is not None and charge_mb > self.cap_memory_mb

    @contextmanager
    def slot(self, *, memory_mb: Optional[int] = None, cores: int = 1) -> Iterator[None]:
        """Hold one run's worth of the budget for the duration of the block.

        Blocks until the run fits, so a caller never has to poll and a waiter is
        woken by the release that makes room. The ``finally`` release runs on an
        exception too — a VM that failed still gives its cores back — and every
        waiter is woken, so an exception cannot wedge the invocation.

        [VANTA-WHY] M3 item X2 — the V7 rule, carried from the ROW charge to the
        SUITE-RUN reservation. A run that records no ``memory_mb`` still boots a VM,
        at ``shuru.DEFAULT_RUN_MEMORY_MB`` (``DEFAULT_VM_MEMORY_MB``), so it must
        reserve that size rather than zero; charging ``None`` as ``0`` let a
        concurrent ``prepare`` run more VMs than the memory cap allowed and report a
        zero peak. An explicitly supplied value is unchanged, including the clamp of
        a negative one to ``0`` — which :meth:`_fits` reads as "do not account memory
        for this run", exactly as before.
        """
        want_cores = max(1, int(cores))
        want_memory = (
            DEFAULT_VM_MEMORY_MB if memory_mb is None else max(0, int(memory_mb))
        )
        with self._cond:
            while not self._fits(want_cores, want_memory):
                self._cond.wait()
            self._used_cores += want_cores
            self._used_memory_mb += want_memory
            self.peak_cores = max(self.peak_cores, self._used_cores)
            self.peak_memory_mb = max(self.peak_memory_mb, self._used_memory_mb)
        try:
            yield
        finally:
            with self._cond:
                self._used_cores -= want_cores
                self._used_memory_mb -= want_memory
                self._cond.notify_all()

    def _fits(self, cores: int, memory: int) -> bool:
        if self._used_cores + cores > self.cores:
            return False
        if memory == 0 or self.cap_memory_mb is None:
            return True
        if memory > self.cap_memory_mb:
            # A single run larger than the whole memory budget still has to run —
            # refusing would be a scheduler that cannot start the one job it was
            # given. It runs alone, which is the closest thing to "fits".
            return self._used_cores == 0
        return self._used_memory_mb + memory <= self.cap_memory_mb

    def record(self) -> dict:
        """The budget as a row records it — what was allowed, and what happened."""
        return {
            "concurrent": self.concurrent,
            "jobs": self.jobs,
            "budget": {
                "cores": self.cores,
                "cap_cores": self.cap_cores,
                "memory_mb": self.cap_memory_mb,
            },
            "host_cores": host_cores(),
            "host_load": self.host_load,
            "peak_cores": self.peak_cores,
            "peak_memory_mb": self.peak_memory_mb,
        }


# ============================================================================
# HostSampler — the batch-level time series (M3 item S6)
# ============================================================================
#
# [VANTA-WHY] Nathan: "For memory, do not attribute per row: on macOS the VMs run
# inside Virtualization XPC service processes that may not be children of the
# row. Instead a batch-level sampler records host-wide used memory and the total
# live VM count over time; calibration uses those totals." So the per-row number
# (`peak_sandboxes` on the row) and the host-wide numbers (here) are deliberately
# different measurements of the same phenomenon, taken from the two places that
# can each see one of them.
SAMPLE_INTERVAL_SECS = 5.0

HOST_SAMPLES_NAME = "host_samples.jsonl"

# One live microVM is hosted by one `shuru run` process (the Virtualization
# framework runs inside it), so counting those processes host-wide is counting
# VMs — the only view the host has, since `shuru` has no `ps`/`list`.
_SHURU_RUN = re.compile(r"(?:^|/|\s)shuru\s+run(?:\s|$)")

# The three page classes macOS reports as resident application memory. Activity
# Monitor's "Memory Used" is these plus wired and compressed, and it is the figure
# a calibration reader can compare against the charge.
_VM_STAT_FIELDS = ("Anonymous pages", "Pages wired down", "Pages occupied by compressor")
_VM_STAT_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_LINE = re.compile(r'^"?(?P<name>[^":]+)"?:\s+(?P<count>\d+)\.')


def host_used_memory_mb() -> Optional[int]:
    """Host-wide USED physical memory in MB, best effort, or ``None``.

    Read straight from ``vm_stat`` so the sampler does not depend on a
    third-party module being installed on the host. It never raises: any shape it
    does not recognise returns ``None`` rather than a wrong number, because a
    calibration sample that lies is worse than one that is missing.
    """
    try:
        completed = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            # [VANTA-WHY] M3 item V4 — `vm_stat`'s output is decoded, and a byte
            # that is not valid UTF-8 raises `UnicodeDecodeError`, which is a
            # `ValueError` and so slips past the `except` below and takes the
            # sampler's thread down. `errors="replace"` turns such a byte into
            # U+FFFD, which this parser already treats as an unrecognised line:
            # the reads stay best-effort, which is the contract of both probes.
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    page_size = 4096
    used_pages = 0
    seen = False
    for line in completed.stdout.splitlines():
        page_size_match = _VM_STAT_PAGE_SIZE.search(line)
        if page_size_match:
            page_size = int(page_size_match.group(1))
            continue
        match = _VM_STAT_LINE.match(line.strip())
        if match and match.group("name") in _VM_STAT_FIELDS:
            used_pages += int(match.group("count"))
            seen = True
    if not seen:
        return None
    return (used_pages * page_size) // (1024 * 1024)


def host_live_vm_count() -> Optional[int]:
    """How many shuru microVMs are live on this host right now, or ``None``."""
    try:
        completed = subprocess.run(
            ["ps", "-axo", "command="],
            capture_output=True,
            text=True,
            # [VANTA-WHY] M3 item V4 — same as `vm_stat` above: a process command
            # line can hold any byte, and decoding it strictly raises a
            # `UnicodeDecodeError` (a `ValueError`) that the `except` below does
            # not catch. Replacing undecodable bytes keeps the count best-effort.
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return sum(1 for line in completed.stdout.splitlines() if _SHURU_RUN.search(line))


class HostSampler:
    """M3 item S6 — a batch-level time series of host memory and live VMs.

    One daemon thread appends one JSON line per interval to ``path``. It is
    deliberately fire-and-forget: a host probe that fails writes ``null`` for its
    field, and the whole sampler is best-effort, because everything it produces is
    calibration data and no verdict reads it — a sampler must never be able to
    take a batch down.
    """

    def __init__(
        self, path: Path, *, interval_secs: float = SAMPLE_INTERVAL_SECS
    ) -> None:
        self.path = Path(path)
        self.interval_secs = float(interval_secs)
        self.samples: list[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def started(self) -> bool:
        """Whether the sampling thread is running."""
        return self._thread is not None

    def start(self) -> "HostSampler":
        """Start sampling. Idempotent, so every caller may ask for it."""
        if self._thread is not None:
            return self
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="vanta-host-sampler", daemon=True
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_secs)

    def sample_once(self) -> dict:
        """Take one sample, append it, and return it (also used by the tests)."""
        sample = {
            "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "used_memory_mb": host_used_memory_mb(),
            "live_vms": host_live_vm_count(),
            "host_load": host_load(),
        }
        self.samples.append(sample)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.flush()
        except OSError:
            pass
        return sample

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.interval_secs + 1.0)
        self._thread = None

    def peak_used_memory_mb(self) -> Optional[int]:
        values = [
            sample["used_memory_mb"]
            for sample in self.samples
            if sample["used_memory_mb"] is not None
        ]
        return max(values) if values else None

    def peak_live_vms(self) -> Optional[int]:
        values = [
            sample["live_vms"]
            for sample in self.samples
            if sample["live_vms"] is not None
        ]
        return max(values) if values else None

    def record(self) -> dict:
        """What the batch keeps about its own sampling, beside the sample file."""
        return {
            "path": str(self.path),
            "interval_secs": self.interval_secs,
            "samples": len(self.samples),
            "peak_used_memory_mb": self.peak_used_memory_mb(),
            "peak_live_vms": self.peak_live_vms(),
        }


def read_host_samples(
    path: Path, *, interval_secs: float = SAMPLE_INTERVAL_SECS
) -> Optional[dict]:
    """The batch's sampler summary, re-derived from its own jsonl, or ``None``.

    [VANTA-WHY] M3 item S6 — ``run_digest.json`` is a view of the batch directory
    and nothing else, so a report regenerated later (``report``) has to describe
    the same host samples the run recorded. Reading the jsonl back, rather than
    requiring the sampler object, is what keeps the run's block and the report's
    block the same block. Its shape mirrors :meth:`HostSampler.record` exactly.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    samples: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line, exactly as `Manifest.records` drops one
    if not samples:
        return None
    used = [s["used_memory_mb"] for s in samples if s.get("used_memory_mb") is not None]
    vms = [s["live_vms"] for s in samples if s.get("live_vms") is not None]
    return {
        "path": str(path),
        "interval_secs": float(interval_secs),
        "samples": len(samples),
        "peak_used_memory_mb": max(used) if used else None,
        "peak_live_vms": max(vms) if vms else None,
    }

