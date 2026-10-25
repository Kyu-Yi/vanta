"""The record contract between ``prepare`` and ``run`` (design §1.6, §7.2).

``prepare`` writes ``prepare.json``; ``run`` reads it back with
``runner.load_prepared``. Those two halves used to be written by hand, one in each
module, and they drifted. ``IssueText`` grew ``leak``/``failure_kinds``/
``needs_decision`` and the reader was not updated, so the first smoke run died with

    TypeError: IssueText.__init__() missing 3 required positional arguments:
    'leak', 'failure_kinds', and 'needs_decision'

which names neither the field nor the batch it came from — the two things a reader
needs in order to fix it.

[VANTA-WHY] Every dataclass ``prepare.json`` stores now owns one
``to_dict``/``from_dict`` pair, and both halves go through :func:`field_of`. A
writer cannot add a field the reader ignores: the round-trip test reads a whole
batch back and compares every field. And a field that goes missing is reported as
*the field* and *the record*, not as a constructor ``TypeError``.

The ``where`` a caller threads through is the record's location — for
``prepare.json`` the bug and the batch directory — because the dataclass on its own
cannot say which file it was reading, and that is the first thing a reader needs.
"""

from __future__ import annotations

from typing import Any, Mapping


class RecordError(ValueError):
    """A stored record does not match the dataclass that names it."""


def _origin(owner: str, where: str) -> str:
    return f"{owner} in {where}" if where else owner


def field_of(
    payload: Any,
    key: str,
    *,
    owner: str,
    where: str = "",
) -> Any:
    """``payload[key]``, or a :class:`RecordError` naming the field and the record."""
    if not isinstance(payload, Mapping):
        raise RecordError(
            f"{_origin(owner, where)}: expected a JSON object, got "
            f"{type(payload).__name__}. The record was written by a different "
            "version of the harness than the one reading it."
        )
    if key not in payload:
        raise RecordError(
            f"{_origin(owner, where)}: missing field {key!r}. The record was written "
            "by a different version of the harness than the one reading it; re-run "
            "the stage that wrote it rather than editing the file by hand."
        )
    return payload[key]


def strings_of(payload: Any, key: str, *, owner: str, where: str = "") -> list[str]:
    """``payload[key]`` as a list of ``str``.

    The shape is checked as well as the presence: a writer that recorded a bare
    string where the reader expects a list would otherwise be read as a list of
    characters, which is a silent corruption rather than a loud one.
    """
    value = field_of(payload, key, owner=owner, where=where)
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise RecordError(
            f"{_origin(owner, where)}: field {key!r} is "
            f"{type(value).__name__}, not a list"
        )
    return [str(item) for item in value]


def map_of(payload: Any, key: str, *, owner: str, where: str = "") -> dict[str, str]:
    """``payload[key]`` as a ``str -> str`` mapping.

    The counterpart of :func:`strings_of` for the fields whose value is a record
    rather than a list (``builds``): the shape is checked so that a writer which
    recorded a list where the reader expects a mapping arrives as a
    :class:`RecordError` naming the field, not as an ``AttributeError`` from inside
    the reader.
    """
    value = field_of(payload, key, owner=owner, where=where)
    if not isinstance(value, Mapping):
        raise RecordError(
            f"{_origin(owner, where)}: field {key!r} is "
            f"{type(value).__name__}, not a mapping"
        )
    return {str(k): str(v) for k, v in value.items()}
