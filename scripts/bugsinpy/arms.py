"""The arm registry — one table for every arm the harness can run (§4.2, §7.1).

An **arm** is one experimental condition: the switch settings, the attempt budget,
and the name a row carries. Until M3 item 8 the harness had two, spelled ``baseline``
and ``vanta``, and the name was DERIVED from the switches by
``pipeline::driver::benchmark_config_name`` — which is what made the recorded config
unable to disagree with the config that ran.

[VANTA-WHY] Item 11's B0 is why this table exists rather than a third ``if`` in that
function. B0 is "attacker off, one Creator call"; B1 (``baseline``) is "attacker off"
too. The two differ ONLY in the attempt budget, which ``benchmark_config_name``
cannot see, so the arm name has to be *stated* — ``VANTA_ARM`` — and this table is
what states it. The name and the switches still cannot drift apart: the harness
writes both from one entry here, and ``env_flag``-style reading plus
``arm_switch_signature`` in ``driver.rs`` refuses a stated name the switches
contradict.

Two tables, deliberately different things:

``ARMS``       canonical arm name to its switch settings, attempt budget, allowed
               modes, and the ``config`` string its rows carry. This is what the
               row's ``arm`` column records and what ``VANTA_ARM`` carries.
``SPELLINGS``  what a split's ``configs`` list may write, to the arm that runs in
               each harness mode. A mode that is absent is refused rather than
               scheduled, which is what makes ``b1`` repair-only and ``suite``
               review-only.

[VANTA-WHY] ``baseline`` is ONE spelling covering TWO arms — ``b1`` in repair,
``suite`` in review — because that is what the legacy runs did: one ``config``
column, two modes, and the pair §7.1 compares. Giving the halves their own
``config`` column values would have changed the job key of every row already on
disk, and the key is the resume cursor.

[VANTA-WHY] Two spellings that normalise to one ``config`` are refused
(:func:`resolve_configs`). The job key has no arm column (``manifest.job_key``), so
two arms under one config name would share a key: the second would be skipped as the
first on a resume, and the manifest would hold one experiment's row under the other's
name.

M3.5's arms (``b3-v1``, ``b3-v2-*``, the ablations) are added by appending to
``ARMS`` and, where they are split-addressable, to ``SPELLINGS`` — the rest of the
harness reads both tables and needs no change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

REPAIR = "repair"
REVIEW = "review"


class ArmError(RuntimeError):
    """The registry cannot answer: an unknown name, or a combination it refuses."""


@dataclass(frozen=True)
class Arm:
    """One experimental condition (§4.2)."""

    name: str
    """Canonical name: what the row's ``arm`` column records, and what the harness
    passes as ``VANTA_ARM``."""

    config: str
    """What the row's ``config`` column records.

    The legacy arms keep the name every row already on disk carries (``baseline``,
    ``vanta``), so their job keys — built from ``config``, never from ``arm`` — stay
    byte-identical and an existing batch still resumes. The arms added after them are
    their own config, which is how their keys differ from every legacy key."""

    modes: tuple[str, ...]
    """The harness modes this arm runs.

    A mode outside the list is refused (:func:`arm_for`) rather than scheduled. B0 is
    repair-only: a B0 review row would be byte-identical to today's baseline review —
    the attacker is off in both — so it could only collide on the key with ``suite``
    and would add no evidence."""

    attacker: bool
    planner: bool
    bate: bool

    max_attempts: Optional[int]
    """The attempt budget this arm pins, or ``None`` for the invocation's own.

    B0 pins 1: "one Creator call, no feedback" is the arm's definition (M3 plan item
    11), not a parameter a caller may vary. Every other arm runs the ``--max-attempts``
    the invocation asked for, which is 4 — the binary's own default."""


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        # M3 item 11 — B0: one Creator call, no feedback, no Attacker.
        Arm("b0", "b0", (REPAIR,), False, False, True, 1),
        # B1 — today's `baseline` repair.
        Arm("b1", "baseline", (REPAIR,), False, False, True, None),
        # M3 item 8 — B2: the Attacker runs, BATE does not admit-test.
        Arm("b2", "b2", (REPAIR, REVIEW), True, False, False, None),
        # B3 — today's `vanta`, both modes.
        Arm("b3", "vanta", (REPAIR, REVIEW), True, False, True, None),
        # The review half of today's `baseline`: attacker off, reviewing a candidate.
        Arm("suite", "baseline", (REVIEW,), False, False, True, None),
    )
}

_SPELLINGS: dict[str, dict[str, str]] = {
    "baseline": {REPAIR: "b1", REVIEW: "suite"},
    "b1": {REPAIR: "b1"},
    "suite": {REVIEW: "suite"},
    "vanta": {REPAIR: "b3", REVIEW: "b3"},
    "b3": {REPAIR: "b3", REVIEW: "b3"},
    "b0": {REPAIR: "b0"},
    "b2": {REPAIR: "b2", REVIEW: "b2"},
}

CONFIGS: tuple[str, ...] = tuple(_SPELLINGS)
"""Every spelling a split's ``configs`` list may carry, in table order.

``vanta_planner`` is deliberately absent. It is §4.4's placeholder — its runs return
``InfraFailure`` — so a split naming it can only produce rows that fail the wave
gate, and ``split.validate`` refuses it before the batch starts."""


def arm(name: str) -> Arm:
    """One arm by canonical name."""
    entry = ARMS.get(name)
    if entry is None:
        raise ArmError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return entry


def modes_of(written: str) -> dict[str, str]:
    """The ``{mode: arm name}`` table for a split spelling, or a refusal."""
    entry = _SPELLINGS.get(written)
    if entry is None:
        raise ArmError(f"unknown config {written!r}; known: {list(CONFIGS)}")
    return entry


def config_of(written: str) -> str:
    """The ``config`` column value a spelling's rows carry.

    Both arms behind ``baseline`` record the same config, which is what keeps a
    ``baseline`` job key byte-identical to a legacy one.
    """
    configs = {ARMS[name].config for name in modes_of(written).values()}
    if len(configs) != 1:
        raise ArmError(
            f"spelling {written!r} spans configs {sorted(configs)}: one spelling must "
            "normalise to one config name, or its two modes would take two job keys"
        )
    return configs.pop()


def arm_for(config: str, mode: str) -> str:
    """The canonical arm a persisted ``(mode, config)`` pair runs, or a refusal."""
    modes = modes_of(config)
    name = modes.get(mode)
    if name is None:
        raise ArmError(
            f"{config!r} does not run {mode!r}; it is a "
            f"{'/'.join(sorted(modes))}-only arm"
        )
    return name


def arm_for_config(config: str) -> Arm:
    """The arm behind a config name, for the ``(attacker, planner)`` switch pair.

    A spelling covering two arms (`baseline`) is answered from its first mode, which
    is sound because the arms it covers agree about the switches: ``b1`` and
    ``suite`` are both the attacker-off path, and that agreement is asserted by
    ``test_bugsinpy.py::TestArms::test_the_two_arms_of_one_spelling_agree_about_the_switches``.
    """
    return ARMS[arm_for(config, sorted(modes_of(config))[0])]


def arm_max_attempts(name: str, default: int) -> int:
    """The attempt budget one arm runs with: its pin, or the invocation's own."""
    pinned = arm(name).max_attempts
    return default if pinned is None else pinned


def resolve_configs(spellings: Sequence[str]) -> list[str]:
    """A split's ``configs`` list as ``config`` column values, or a refusal.

    See the module docstring: two spellings that normalise to one config are refused,
    because the job key has no arm column and the two would share a cursor entry.
    """
    out: list[str] = []
    seen: dict[str, str] = {}
    for written in spellings:
        config = config_of(written)  # raises for an unknown spelling
        if config in seen:
            raise ArmError(
                f"configs {seen[config]!r} and {written!r} both normalise to "
                f"{config!r}: two arms under one config name would share a job key"
            )
        seen[config] = written
        out.append(config)
    return out


def phases_for(spellings: Sequence[str]) -> list[tuple[str, str]]:
    """Every repair phase, then every review phase — D13's order, for any arm list.

    Mode-major, so a config's review rows can only be scheduled after every config's
    repairs: a review pass reads the repair rows the pool is built from. The order is
    built here rather than checked against a constant, so it cannot drift from the
    list it describes.

    The second element is the **spelling** the split wrote, not the ``config`` value
    it normalises to. ``b1`` and ``suite`` share the config ``baseline`` but run one
    phase each, so the spelling — not the config — is what says which phase exists.
    """
    out: list[tuple[str, str]] = []
    for mode in (REPAIR, REVIEW):
        for written in spellings:
            if mode in modes_of(written):
                out.append((mode, written))
    return out


def arm_of(mode: str, config: str) -> str:
    """The canonical arm for a persisted ``(mode, config)`` pair — the backfill.

    [VANTA-WHY] Rows written before the ``arm`` column existed carry no arm, and the
    loader derives it here so an old row and a new one group under the same arm
    instead of splitting one experiment into a "recorded" and an "unrecorded" half.

    A pair no entry describes falls back to the config string: the row is then its
    own arm, and inventing a canonical name for an experiment nobody registered would
    be a claim about a run no entry describes.
    """
    try:
        return arm_for(config, mode)
    except ArmError:
        return config


def row_arm(record: Mapping[str, Any]) -> str:
    """A manifest row's arm: its own field, or backfilled from ``(mode, config)``."""
    recorded = record.get("arm")
    if isinstance(recorded, str) and recorded:
        return recorded
    return arm_of(str(record.get("mode") or ""), str(record.get("config") or ""))
