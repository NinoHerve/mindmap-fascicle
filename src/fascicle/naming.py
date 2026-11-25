"""BIDS-style naming for derivatives: any suffix, folder and extension.

`dm.bids_namer` covers raw BIDS through pybids' built-in patterns, which
only know the raw spec's suffixes, folders and extensions. For your own
derivatives, `derivative_namer` builds a pybids path pattern from a short
declaration and lets pybids fill it:

    namer = derivative_namer("mask", datatype="anat", extensions=(".nii.gz", ".json"))
    namer(root, "01", session="a", space="MNI152", desc="brain")
    # root/sub-01/ses-a/anat/sub-01_ses-a_space-MNI152_desc-brain_mask.nii.gz

Entities are written in the BIDS spec's order whatever order you pass them
in. An entity, suffix, folder or extension the namer wasn't declared with
raises `ValueError` -- nothing is silently dropped -- and so do values the
spec doesn't allow (e.g. `desc="brain-mask"`, `run="one"`, `hemi="lh"`).
"""

import re
from pathlib import Path
from typing import Any, Callable

from bids.layout.writing import build_path

# Every BIDS entity except subject (always present), in the spec's filename
# order: (keyword argument, filename key, value format, allowed values).
# Keyword arguments are pybids' entity names where pybids has the entity
# (session, acquisition, desc, ...), so raw and derivative namers take the
# same arguments; otherwise the filename key. Formats come from the spec:
# "label" is alphanumeric, "index" is a non-negative integer.
# tests/test_naming.py checks this table against the installed BIDS schema.
BIDS_ENTITIES: tuple[tuple[str, str, str, tuple[str, ...] | None], ...] = (
    ("template", "tpl", "label", None),
    ("session", "ses", "label", None),
    ("cohort", "cohort", "label", None),
    ("sample", "sample", "label", None),
    ("task", "task", "label", None),
    ("tracksys", "tracksys", "label", None),
    ("acquisition", "acq", "label", None),
    ("nucleus", "nuc", "label", None),
    ("volume", "voi", "label", None),
    ("ceagent", "ce", "label", None),
    ("tracer", "trc", "label", None),
    ("staining", "stain", "label", None),
    ("reconstruction", "rec", "label", None),
    ("direction", "dir", "label", None),
    ("run", "run", "index", None),
    ("modality", "mod", "label", None),
    ("echo", "echo", "index", None),
    ("flip", "flip", "index", None),
    ("inv", "inv", "index", None),
    ("mt", "mt", "label", ("on", "off")),
    ("part", "part", "label", ("mag", "phase", "real", "imag")),
    ("proc", "proc", "label", None),
    ("hemi", "hemi", "label", ("L", "R")),
    ("space", "space", "label", None),
    ("split", "split", "index", None),
    ("recording", "recording", "label", None),
    ("chunk", "chunk", "index", None),
    ("atlas", "atlas", "label", None),
    ("segmentation", "seg", "label", None),
    ("scale", "scale", "label", None),
    ("res", "res", "label", None),
    ("den", "den", "label", None),
    ("label", "label", "label", None),
    ("desc", "desc", "label", None),
)

def subject_label(subject) -> str:
    """Return the bare subject label: "sub-01" and "01" both give "01"."""
    return str(subject).removeprefix("sub-")


_FORMATS = {"label": re.compile(r"[0-9a-zA-Z+]+"), "index": re.compile(r"[0-9]+")}
_SPEC = {name: (key, fmt, allowed) for name, key, fmt, allowed in BIDS_ENTITIES}


def _resolve_entities(
    entities: tuple[str, ...] | None, extra_entities: tuple[str, ...]
) -> list[tuple[str, str, str, tuple[str, ...] | None]]:
    """Return the allowed entities as table rows, in filename order."""
    if entities is not None:
        unknown = sorted(set(entities) - set(_SPEC) - set(extra_entities))
        if unknown:
            raise ValueError(
                f"Unknown BIDS entities {unknown}. Spec entities: {list(_SPEC)}. "
                "Declare your own with extra_entities=."
            )

    keys = {key for key, _, _ in _SPEC.values()}
    for extra in extra_entities:
        if extra in _SPEC or extra in keys or extra == "subject":
            raise ValueError(f"extra entity '{extra}' clashes with a BIDS spec entity.")
        if not _FORMATS["label"].fullmatch(extra) or "+" in extra:
            raise ValueError(f"extra entity name '{extra}' must be alphanumeric.")

    rows = [row for row in BIDS_ENTITIES if entities is None or row[0] in entities]
    # Custom entities go just before desc, which stays last by convention.
    customs = [(name, name, "label", None) for name in extra_entities]
    if rows and rows[-1][0] == "desc":
        return rows[:-1] + customs + rows[-1:]
    return rows + customs


def _check_value(row: tuple[str, str, str, tuple[str, ...] | None], value) -> str:
    """Return `value` as text, raising if the spec doesn't allow it for this entity."""
    ent, _, fmt, allowed = row
    text = str(value)
    if allowed is not None and text not in allowed:
        raise ValueError(f"{ent}={value!r} must be one of {list(allowed)}.")
    if not _FORMATS[fmt].fullmatch(text):
        kind = "alphanumeric" if fmt == "label" else "a non-negative integer"
        raise ValueError(f"{ent}={value!r} must be {kind} (BIDS {fmt}).")
    return text


def _declare(
    entities: tuple[str, ...] | None,
    extra_entities: tuple[str, ...],
    fixed: dict[str, Any] | None,
) -> tuple[list[tuple[str, str, str, tuple[str, ...] | None]], dict[str, str]]:
    """Return (all entity rows in filename order, validated fixed values)."""
    fixed = dict(fixed or {})
    unknown = sorted(set(fixed) - set(_SPEC) - set(extra_entities))
    if unknown:
        raise ValueError(
            f"Fixed entities {unknown} are neither BIDS spec entities nor declared "
            "in extra_entities=."
        )
    if entities is not None:
        entities = tuple(entities) + tuple(k for k in fixed if k not in entities)

    rows = _resolve_entities(entities, extra_entities)
    by_name = {row[0]: row for row in rows}
    return rows, {ent: _check_value(by_name[ent], value) for ent, value in fixed.items()}


def bids_pattern(
    suffix: str,
    datatype: str | None = None,
    extensions: tuple[str, ...] | None = (".nii.gz",),
    entities: tuple[str, ...] | None = None,
    extra_entities: tuple[str, ...] = (),
    fixed: dict[str, Any] | None = None,
) -> str:
    """Return the pybids path pattern `derivative_namer` fills.

    Parameters
    ----------
    suffix: str
        BIDS suffix, e.g. "mask", "conn".
    datatype: str, optional
        Folder under the subject/session folder, e.g. "anat". None writes
        files directly in `sub-XX[/ses-YY]/`.
    extensions: tuple[str, ...], optional
        Allowed extensions; the first is the default, others are chosen
        with `extension=` (e.g. a ".json" sidecar). Empty or None builds
        extensionless stems.
    entities: tuple[str, ...], optional
        Entities this datatype accepts (keyword names from
        `BIDS_ENTITIES`). None accepts every spec entity.
    extra_entities: tuple[str, ...]
        Your own entities, written as `_<name>-<value>` just before `desc`.
    fixed: dict, optional
        Entities with a value fixed for this datatype, always written and
        never passed by the caller, e.g. `{"signal": "virtual"}` (a custom
        entity, so also listed in `extra_entities`).
    """
    for part, value in (("suffix", suffix), ("datatype", datatype)):
        if value is not None and (not value or any(c in value for c in "{}[]<>|/_ ")):
            raise ValueError(f"{part} {value!r} can't be empty or contain {{}}[]<>|/_ or spaces.")

    rows, fixed = _declare(entities, tuple(extra_entities), fixed)

    # suffix, datatype and fixed entities are written literally (they're
    # fixed per namer); pybids' `{entity<...>|default}` syntax rejects
    # hyphens in defaults, which suffixes such as "i-ave" would need.
    def part(ent: str, key: str) -> str:
        return f"_{key}-{fixed[ent]}" if ent in fixed else f"[_{key}-{{{ent}}}]"

    folder = "sub-{subject}"
    if "session" in fixed:
        folder += f"/ses-{fixed['session']}"
    elif any(ent == "session" for ent, *_ in rows):
        folder += "[/ses-{session}]"
    if datatype is not None:
        folder += f"/{datatype}"

    name = "sub-{subject}" + "".join(part(ent, key) for ent, key, *_ in rows)
    name += f"_{suffix}"
    if extensions:
        name += f"{{extension<{'|'.join(extensions)}>|{extensions[0]}}}"

    return f"{folder}/{name}"


def derivative_namer(
    suffix: str,
    datatype: str | None = None,
    extensions: tuple[str, ...] | None = (".nii.gz",),
    entities: tuple[str, ...] | None = None,
    extra_entities: tuple[str, ...] = (),
    fixed: dict[str, Any] | None = None,
) -> Callable[..., Path]:
    """Return a namer building BIDS-style derivative paths for `suffix`.

    The namer has the `Datatype` signature `namer(root, subject, **entities)`
    and returns `root / <BIDS-style relative path>`. See `bids_pattern` for
    the parameters.

    Raises (from the namer)
    ------
    ValueError
        If an entity, extension or folder wasn't declared (or is fixed), or
        a value doesn't match the spec's format or allowed values.
    """
    extra_entities = tuple(extra_entities)
    pattern = bids_pattern(suffix, datatype, extensions, entities, extra_entities, fixed)
    all_rows, fixed_values = _declare(entities, extra_entities, fixed)
    rows = {row[0]: row for row in all_rows if row[0] not in fixed_values}

    def namer(root, subject, **values) -> Path:
        unknown = sorted(set(values) - set(rows) - {"extension"})
        if unknown:
            fixed_note = f" Fixed for this datatype: {fixed_values}." if fixed_values else ""
            raise ValueError(
                f"'{suffix}' doesn't take entities {unknown}. Allowed: {list(rows)}.{fixed_note}"
            )
        extension = values.get("extension")
        if extension is not None and extension not in (extensions or ()):
            raise ValueError(
                f"'{suffix}' doesn't take extension {extension!r}. "
                f"Allowed: {list(extensions or [])}."
            )

        for ent, value in values.items():
            if ent in rows and value is not None:
                _check_value(rows[ent], value)

        source = {"subject": subject_label(subject), **values}
        relative = build_path(source, [pattern], strict=True)
        if relative is None:
            raise ValueError(f"Could not name '{suffix}' with {values} (pattern: {pattern}).")
        return Path(root) / relative

    namer.__name__ = f"derivative_namer({suffix!r})"
    namer.pattern = pattern
    return namer
