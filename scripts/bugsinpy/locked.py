"""The one hash-locked artifact form (§2.4 item 2, item 5, item 6).

Every frozen file the harness writes has the same shape: a title line, a
``- key: value`` header, a marker line, a body, and the sha256 of that body in
the header. Three writers use it — ``issue_source`` (the frozen issue report),
``mutants`` (item 5's pool) and ``pool`` (item 6's assembled pool) — and before
this module they were three copies of the same mechanics. A copy is a place for
a fix to land on one side and miss the other, so the mechanics live here once
and each writer supplies only its own title, header lines, marker and body.

What is *not* here is what differs: which keys are required, which sections a
body has, and what each section's hash means. Those stay with the writer that
knows them; this module owns the string handling and the write/verify order that
makes a file its own reader can accept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple, Type

from .split import sha256_text


def fold(value: str) -> str:
    """The reporter's own text: line endings folded, and **only** its edge newlines dropped.

    ``\\r\\n`` and a lone ``\\r`` become ``\\n``, and the newlines the framing
    around a part would otherwise duplicate are dropped; a space, a tab, a run of
    blanks and a line's indent are bytes of the text and are **never** touched.
    ``str.strip()`` also eats a trailing space or the leading indentation of a body
    that opens with a code block, and a code block is source the reporter pasted.
    """
    return value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def canonical(value: str) -> str:
    """The one form a body takes — in the file, and in the hash of it.

    Three rules, and they are the *same* rule for the writer and the reader: line
    endings are folded; the blank lines the framing puts at a section's edges are
    dropped; and a non-empty body ends with exactly one ``\\n``. The first two are
    :func:`fold`; the third is only about the file's shape.

    [VANTA-WHY] The fold is what the first dev freeze got wrong, and that failure
    was silent rather than loud. The API returns text with whatever ``\\r\\n`` the
    author's editor used; a writer that hashed that text as it stood and read it
    back through a universal-newline ``read_text`` would disagree with itself on
    **every** report containing a carriage return. Folding first makes writer and
    reader agree *by rule* rather than by both sides happening to do the same
    thing.

    [VANTA-WHY] The edge rule removes **newlines only, and never whitespace**. A
    body's first and last character are never the ones the writer added as framing;
    dropping newlines recovers the body exactly. Stripping instead also eats a
    trailing space or tab, and that is not framing: it is a byte of the text, and a
    rule that hides it would let a hand-edited artifact verify.
    """
    folded = fold(value)
    return folded + "\n" if folded else ""


def content_hash(body: str) -> str:
    """The header's ``content-sha256`` — the hash of a body in its canonical form."""
    return sha256_text(canonical(body))


def header_block(text: str) -> Tuple[dict[str, str], str]:
    """``(header, the rest)`` — the ``- key: value`` lines, up to the first blank line."""
    lines = text.splitlines(keepends=True)
    header: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("- ") and ":" in line:
            key, _, value = line[2:].partition(":")
            header[key.strip()] = value.strip()
        elif header and not line.strip():
            index += 1
            break
        index += 1
    return header, "".join(lines[index:])


def section(
    text: str, marker: str, *, error: Type[Exception], what: str
) -> Tuple[str, str]:
    """``(before, after)`` the first line equal to ``marker``, that line dropped.

    A marker line rather than a blank one, because a section's text may be empty
    and a reader that split on "the first blank line" could not tell that from a
    truncated file.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return "".join(lines[:index]), "".join(lines[index + 1 :])
    raise error(f"the {what} has no {marker} line — it is not a file this reader may use")


def require_keys(
    header: Mapping[str, str],
    keys: Sequence[str],
    *,
    path: Path,
    error: Type[Exception],
    what: str,
    why: str,
) -> None:
    """Refuse a header that cannot say what it came from."""
    for key in keys:
        if key not in header:
            raise error(
                f"{path}: the header has no {key!r} line; a {what} that cannot "
                f"{why} is not one the reader may use"
            )


def write(path: Path, text: str, *, verify: Callable[..., Any]) -> Path:
    """Render → verify with the reader → write, ``newline=""`` so the bytes hash.

    [VANTA-WHY] The render is verified *before* it is written, through the same
    parser a stage reads with. The first dev freeze wrote artifacts its own reader
    refused, and nothing said so at the time: the refusal surfaced one stage later,
    as a failure on a bug whose file looked fine on disk. Verifying first makes an
    internally inconsistent artifact impossible to create rather than merely
    detectable, and it leaves no file behind when the two do disagree.

    ``newline=""`` because the bytes written are the bytes the reader hashes: a
    platform that translated ``\\n`` on the way out would put every hash off by one
    byte per line.
    """
    target = Path(path)
    verify(text, path=target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return target


def load(path: Path, *, parse: Callable[..., Any], error: Type[Exception], what: str) -> Any:
    """Read a locked file from disk, and verify it (``parse``).

    ``newline=""``: no translation in either direction, so the bytes hashed are the
    bytes written. Reading with universal newlines is what let a reader lose one
    ``\\r`` per line that the writer had hashed.
    """
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()
    except OSError as exc:
        raise error(f"cannot read the {what} {target}: {exc}") from exc
    return parse(text, path=target)
