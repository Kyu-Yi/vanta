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
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

HOST_CORE_RESERVE = 2
"""The cores this scheduler never allocates — the host keeps them for itself.

[VANTA-WHY] The mandate's cap is "host cores − 2". The reserve is what keeps a
batch from starving the machine it runs on: the harness process, the shell that
launched it, and whatever the operator has open all need a core, and a scheduler
that took all of them would make a concurrent batch *slower* than a serial one
by making every process wait for a core.
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
    ) -> None:
        self.jobs = max(1, int(jobs))
        self.cap_cores = (
            host_cores_budget() if cores_cap is None else max(1, int(cores_cap))
        )
        self.cap_memory_mb = (
            host_memory_mb() if memory_cap_mb is None else memory_cap_mb
        )
        self.cores = max(1, min(self.jobs, self.cap_cores))
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

    @contextmanager
    def slot(self, *, memory_mb: Optional[int] = None, cores: int = 1) -> Iterator[None]:
        """Hold one run's worth of the budget for the duration of the block.

        Blocks until the run fits, so a caller never has to poll and a waiter is
        woken by the release that makes room. The ``finally`` release runs on an
        exception too — a VM that failed still gives its cores back — and every
        waiter is woken, so an exception cannot wedge the invocation.
        """
        want_cores = max(1, int(cores))
        want_memory = 0 if memory_mb is None else max(0, int(memory_mb))
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
