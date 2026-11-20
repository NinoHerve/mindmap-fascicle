"""Shared path/extension helpers used by the core manager and reader plugins.

Holds parse_extension, check_file, and a BIDS derivatives path helper,
adapted from mindmap's `mindmap/io/utils.py`.
"""

from pathlib import Path


# Two-part extensions treated as one, e.g. BIDS's compressed images and
# compressed physio/stim tables. Anything else ending in ".gz" stays ".gz".
COMPOUND_EXTENSIONS = (".nii.gz", ".tsv.gz")


def parse_extension(filename: str | Path) -> str:
    """Return the extension of `filename`.

    Treats the two-part extensions in `COMPOUND_EXTENSIONS` (".nii.gz",
    ".tsv.gz") as single extensions rather than just ".gz".
    """
    suffixes = Path(filename).suffixes

    if len(suffixes) >= 2 and "".join(suffixes[-2:]) in COMPOUND_EXTENSIONS:
        return "".join(suffixes[-2:])

    return suffixes[-1] if suffixes else ""


def check_file(
    fname: str | Path,
    extension: list[str] | str | None = None,
    instance: str | None = None,
) -> str | Path:
    """Check that a file exists and, optionally, that it has an expected extension.

    Parameters
    ----------
    fname: str | Path
        File name.
    extension: str | list[str], optional
        Extension(s) to accept (with or without a leading dot).
    instance: str, optional
        "str" or "path" to force the return type. If None, returns a Path.

    Returns
    -------
    fname: str | Path
        The file name, as a Path or as a string per `instance`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If `extension` is given and the file's extension doesn't match.
    """
    f = Path(fname)
    if not f.exists():
        raise FileNotFoundError(f"Following file was not found: {fname}")

    if extension is not None:
        if not isinstance(extension, list):
            extension = [extension]

        extension = [ext if ext.startswith(".") else "." + ext for ext in extension]

        if parse_extension(f) not in extension:
            raise ValueError(f"Invalid file extension for {f}. Expected {extension}")

    if instance == "str":
        return str(f)

    return f


def build_bids_path(layout, subject: str, suffix: str, **kwargs) -> Path:
    """Build a BIDS-compliant path via a pybids layout's `build_path`.

    Works for a raw `BIDSLayout` or any BIDS-derivatives layout. Pure: the
    file needn't exist. `DatasetManager.bids_namer` wraps it for raw-BIDS
    datatypes such as core's "T1w".

    Parameters
    ----------
    layout: bids.BIDSLayout
        The (raw or derivatives) layout to build the path against.
    subject: str
        Subject label.
    suffix: str
        BIDS suffix (e.g. "T1w").
    **kwargs
        Additional BIDS entities (session, task, run, space, desc, ...).

    Raises
    ------
    ValueError
        If pybids cannot resolve a path for the given entities.
    """
    entities = {"subject": subject, "suffix": suffix, **kwargs}

    # strict=True: an entity the matched raw-BIDS pattern doesn't have
    # (desc, space, ...) raises instead of being silently dropped.
    message = (
        f"Could not build a raw BIDS path for suffix '{suffix}' with entities "
        f"{kwargs}. Raw BIDS patterns only cover the raw spec's suffixes, "
        "entities and extensions; for derivatives (desc, space, new suffixes "
        "or extensions) use fascicle.derivative_namer."
    )
    try:
        path = layout.build_path(entities, validate=False, strict=True)
    except ValueError as err:  # pybids raises when no pattern matches
        raise ValueError(message) from err

    if path is None:
        raise ValueError(message)

    return path
