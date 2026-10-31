"""Item 19 — the frozen configuration: one file, one hash, one row field.

[VANTA-WHY] A verdict is a claim about a *configuration*, not about a model. The
same bug run under a different reasoning effort, a different sampling preset or a
different set of audit switches produces a different number, and nothing on disk
used to say which one a row came from: the switches reach ``vanta_m3`` through the
*environment*, and an environment is exactly the thing a later reader cannot
recover. So the M3 configuration lives in one committed file
(``configs/v1-m3.toml``), the file's SHA-256 is written on every manifest row, and
two rows under different files can never be pooled (``analyse.py``).

Three things the file's sections mean, and they are different things on purpose.

``[env]``
    The verdict-affecting settings that are *frozen to a literal*. The harness
    APPLIES these to the child environment, and it REFUSES the run when the
    caller's own environment sets one of them to a different value — unless
    ``--allow-override`` is given, which is recorded on every row. The value
    ``"unset"`` is a frozen value like any other: it means "this variable must be
    absent", so exporting ``VANTA_REASONING_EFFORT`` at all is an override.

``[recorded]``
    The verdict-affecting settings whose value is only knowable at run time —
    which model the endpoint actually served, how many tokens the context
    selector was allowed, which audit flags the invocation set. The file declares
    that they are part of the configuration and says where the observed value
    lives (the row, ``run.summary``); it does not freeze a literal, because a
    literal there would be a claim the harness cannot make true.

``[prereg]`` and ``[prompts]``
    The pre-registration: what the M3 configuration commits to before a run, and
    the content hash of every prompt template. Both are inside the file's bytes,
    so both are inside ``config_hash``.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "v1-m3.toml"

FROZEN_SECTION = "env"
RECORDED_SECTION = "recorded"
PREREG_SECTION = "prereg"
PROMPTS_SECTION = "prompts"

UNSET = "unset"
"""A frozen value meaning "this variable must not be set at all".

[VANTA-WHY] Distinct from "not listed": a variable that is absent from ``[env]`` is
simply not frozen, while one frozen to ``unset`` is frozen *to absence* — its
presence in the caller's environment is an override the run refuses. The two
settings that need it (``VANTA_REASONING_EFFORT``, ``VANTA_PRESENCE_PENALTY``) are
verdict-affecting precisely because they are unset: the model's own default is part
of what the configuration measured, and a caller who sets one measures something
else."""

SCHEMA = 1


class FrozenConfigError(RuntimeError):
    """The frozen configuration refuses the run (a bad file, or an override)."""


@dataclass(frozen=True)
class Override:
    """One caller value that contradicts a frozen setting."""

    name: str
    frozen: str
    actual: str

    def render(self) -> str:
        return f"{self.name}={self.frozen}!={self.actual}"


@dataclass(frozen=True)
class FrozenConfig:
    """A loaded ``configs/*.toml``: its path, the hash of its bytes, its tables."""

    path: Path
    sha256: str
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.path.stem)

    def frozen(self) -> dict[str, str]:
        """``env var -> frozen value`` — the settings a caller cannot contradict."""

        return {str(k): str(v) for k, v in (self.data.get(FROZEN_SECTION) or {}).items()}

    def recorded(self) -> dict[str, str]:
        """``env var -> where the observed value is written`` — declared, not frozen."""

        return {str(k): str(v) for k, v in (self.data.get(RECORDED_SECTION) or {}).items()}

    def prereg(self) -> dict[str, Any]:
        return dict(self.data.get(PREREG_SECTION) or {})

    def prompts(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in (self.data.get(PROMPTS_SECTION) or {}).items()}

    # -- the refusal ------------------------------------------------------
    def overrides(self, environ: Mapping[str, str]) -> list[Override]:
        """Every caller value that contradicts a frozen setting, in sorted order.

        A variable the caller does not set is never an override: the harness
        applies the frozen value, so the run still measures the frozen
        configuration. A variable frozen to ``unset`` IS an override when the
        caller sets it, because the frozen setting there is "absent".
        """

        found: list[Override] = []
        for name, frozen in sorted(self.frozen().items()):
            if name not in environ:
                continue
            actual = str(environ[name])
            if frozen == UNSET:
                found.append(Override(name=name, frozen=UNSET, actual=actual))
            elif actual.strip() != frozen:
                found.append(Override(name=name, frozen=frozen, actual=actual))
        return found

    def applied_env(self, *, waive: Iterable[str] = ()) -> dict[str, str]:
        """The frozen settings to put into the child environment.

        ``unset`` entries contribute nothing (their frozen value is absence, and
        an override that waived one leaves the caller's own value in place — the
        harness only ever adds keys to the inherited environment).
        """

        skipped = {str(name) for name in waive}
        return {
            name: value
            for name, value in sorted(self.frozen().items())
            if value != UNSET and name not in skipped
        }

    # -- the row block ----------------------------------------------------
    def record(
        self, *, allow_override: bool, overrides: Iterable[Override] = ()
    ) -> dict[str, Any]:
        """The block ``build_record`` puts on every row this invocation writes."""

        return {
            "config_hash": self.sha256,
            "config_path": _relative(self.path),
            "config_override": bool(allow_override),
            "config_overrides": [item.render() for item in overrides],
        }


def _relative(path: Path) -> str:
    """A path inside the repo as a repo-relative string, anything else absolute.

    Recorded relative so a batch's rows do not carry the machine's home directory
    (the same rule the rest of the harness's recorded paths follow).
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load(path: Optional[Path | str] = None) -> FrozenConfig:
    """Read one frozen-config file, or refuse with the reason."""
    target = Path(path) if path is not None else CONFIG_PATH
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise FrozenConfigError(f"cannot read the frozen config {target}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise FrozenConfigError(f"{target} is not valid TOML: {exc}") from exc
    _validate(data, target)
    return FrozenConfig(path=target, sha256=hashlib.sha256(raw).hexdigest(), data=data)


def _validate(data: Mapping[str, Any], path: Path) -> None:
    """Refuse a file that cannot mean what the reader below assumes."""

    def table(section: str) -> Mapping[str, Any]:
        value = data.get(section)
        if not isinstance(value, Mapping):
            raise FrozenConfigError(f"{path}: [{section}] must be a table")
        return value

    if data.get("schema") != SCHEMA:
        raise FrozenConfigError(
            f"{path}: schema must be {SCHEMA}, found {data.get('schema')!r}"
        )
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise FrozenConfigError(f"{path}: `name` must be a non-empty string")
    for section in (FROZEN_SECTION, RECORDED_SECTION):
        for key, value in table(section).items():
            if not str(key).startswith("VANTA_"):
                raise FrozenConfigError(
                    f"{path}: [{section}] key {key!r} is not a VANTA_ variable; the "
                    "file describes this harness's own configuration, and a bare "
                    "name here would look frozen while nothing could read it"
                )
            if not isinstance(value, str) or not value:
                raise FrozenConfigError(
                    f"{path}: [{section}] {key} must be a non-empty string"
                )
    table(PREREG_SECTION)
    for key, value in table(PROMPTS_SECTION).items():
        if not isinstance(value, str) or not value:
            raise FrozenConfigError(
                f"{path}: [prompts] {key} must be a non-empty string"
            )


def resolve(
    frozen: FrozenConfig,
    *,
    environ: Mapping[str, str],
    allow_override: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """``(row block, environment to apply)`` for one invocation, or a refusal.

    Both halves come from one place so a caller cannot record an override it did
    not waive, or waive one it did not record.
    """
    found = frozen.overrides(environ)
    if found and not allow_override:
        raise FrozenConfigError(
            f"{frozen.path} freezes "
            + ", ".join(item.render() for item in found)
            + ". The caller's environment contradicts the frozen configuration, so "
            "this run would not measure the configuration its rows claim. Unset the "
            "variable(s), or pass --allow-override to run anyway — the waiver is "
            "then recorded on every row and the batch cannot be pooled with one "
            "that did not need it."
        )
    waived = found if allow_override else []
    return (
        frozen.record(allow_override=allow_override, overrides=waived),
        frozen.applied_env(waive=[item.name for item in waived]),
    )
