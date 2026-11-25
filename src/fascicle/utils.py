"""Shared path/extension helpers used by the core manager and reader plugins.

Holds parse_extension, check_file and a BIDS path helper, adapted from
mindmap's `mindmap/io/utils.py`, plus find_sidecar (BIDS inheritance).
"""

from pathlib import Path

from bids.layout import parse_file_entities


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


# Entities describing the file itself rather than what it's about; ignored
# when matching a sidecar to a recording.
_NOT_MATCHED = ("datatype", "suffix", "extension")


def find_sidecar(
    recording: str | Path, suffix: str, extensions: tuple[str, ...] | str
) -> Path:
    """Return the sidecar `suffix` that applies to `recording`, by BIDS inheritance.

    Walks from the recording's folder up to the dataset root: the first
    folder holding `dataset_description.json`, or else the folder above the
    outermost `sub-*` folder. Outside any BIDS layout (no `sub-*` parent),
    only the recording's own folder is searched. At each level, a candidate
    is a `*_<suffix><ext>` file whose entities are all also entities of the
    recording, with the same values. The nearest level wins; within a level,
    the candidate with the most entities wins. e.g. for
    `sub-01/ses-meg/meg/sub-01_ses-meg_task-face_run-01_meg.fif`,
    `sub-01/ses-meg/sub-01_ses-meg_task-face_channels.tsv` applies.

    Raises
    ------
    FileNotFoundError
        If no sidecar applies.
    ValueError
        If two candidates at the nearest level are equally specific.
    """
    if isinstance(extensions, str):
        extensions = (extensions,)
    recording = Path(recording)

    def entities(path: Path) -> dict:
        parsed = parse_file_entities(str(path))
        return {k: v for k, v in parsed.items() if k not in _NOT_MATCHED}

    own = entities(recording)
    endings = tuple(f"_{suffix}{ext}" for ext in extensions)

    for folder in _inheritance_folders(recording):
        candidates = []
        for path in folder.iterdir():
            if path == recording or not path.is_file() or not path.name.endswith(endings):
                continue
            theirs = entities(path)
            if all(key in own and own[key] == value for key, value in theirs.items()):
                candidates.append((len(theirs), path))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            best = [path for n, path in candidates if n == candidates[0][0]]
            if len(best) > 1:
                names = sorted(p.name for p in best)
                raise ValueError(
                    f"Several '{suffix}' sidecars apply equally to {recording.name}: {names}."
                )
            return best[0]

    raise FileNotFoundError(f"No '{suffix}' sidecar ({list(extensions)}) applies to {recording}.")


def _inheritance_folders(recording: Path) -> list[Path]:
    """Return the folders `find_sidecar` searches, nearest first.

    From the recording's folder up to the dataset root: the first parent
    holding `dataset_description.json`, or else the folder above the
    outermost `sub-*` folder. Outside any BIDS layout, only the recording's
    own folder.
    """
    parents = list(recording.parents)
    for i, folder in enumerate(parents):
        if (folder / "dataset_description.json").exists():
            return parents[: i + 1]

    subjects = [i for i, folder in enumerate(parents) if folder.name.startswith("sub-")]
    if subjects and subjects[-1] + 1 < len(parents):
        return parents[: subjects[-1] + 2]
    return parents[:1]
