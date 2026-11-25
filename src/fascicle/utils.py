"""Shared path/extension helpers used by the core manager and reader plugins.

Holds parse_extension, check_file and a BIDS path helper, adapted from
mindmap's `mindmap/io/utils.py`, plus find_sidecar (BIDS inheritance) and
find_padded (the same file under another zero-padding).
"""

from pathlib import Path

from .naming import BIDS_ENTITIES, subject_label  # noqa: F401 - re-exported


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


# Metadata files the BIDS inheritance principle applies to. Data files
# (.nii.gz, .fif, ...) never inherit.
INHERITABLE_EXTENSIONS = (".json", ".tsv", ".bval", ".bvec")


def find_sidecar(
    recording: str | Path, suffix: str, extensions: tuple[str, ...] | str
) -> Path:
    """Return the sidecar `suffix` that applies to `recording`, by BIDS inheritance.

    Walks from the recording's folder up to the dataset root: the first
    folder holding `dataset_description.json`, or else the folder above the
    outermost `sub-*` folder. Outside any BIDS layout (no `sub-*` parent),
    only the recording's own folder is searched. At each level, a candidate
    is a `*_<suffix><ext>` file whose entities are all also entities of the
    recording, with the same values (index entities such as `run` or
    `split` compared as numbers, so `run-1` matches `run-01`). The nearest level wins; within a level,
    the candidate with the most entities wins. e.g. for
    `sub-01/ses-meg/meg/sub-01_ses-meg_task-face_run-01_meg.fif`,
    `sub-01/ses-meg/sub-01_ses-meg_task-face_channels.tsv` applies. A bare
    `<suffix><ext>` (e.g. `T1w.json`) has no entities, so it applies to every
    recording. `recording` needn't exist, nor its folder: missing folders are
    skipped.

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
        return _filename_entities(path.name)[0]

    own = entities(recording)
    endings = tuple(f"_{suffix}{ext}" for ext in extensions)
    bare = {f"{suffix}{ext}" for ext in extensions}  # dataset-level, e.g. "T1w.json"

    for folder in _inheritance_folders(recording):
        if not folder.is_dir():  # e.g. a built path's datatype folder
            continue
        candidates = []
        for path in folder.iterdir():
            if path == recording or not path.is_file():
                continue
            if not (path.name.endswith(endings) or path.name in bare):
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


# Filename keys of BIDS index entities (run, echo, ...), whose values are
# integers that may be zero-padded to any width: run-1, run-01, run-001.
_INDEX_KEYS = frozenset(key for _, key, fmt, _ in BIDS_ENTITIES if fmt == "index")


def _filename_entities(name: str) -> tuple[dict[str, str | int], str, str]:
    """Split a BIDS file name into (entities, suffix, extension).

    Entities are keyed by their filename key (`ses`, `run`, ...). Index
    values become `int`, so `run-01` and `run-1` give the same entities;
    labels stay strings (`acq-01` and `acq-1` differ).
    """
    extension = parse_extension(name)
    stem = name[: len(name) - len(extension)] if extension else name
    *pairs, suffix = stem.split("_")
    entities: dict[str, str | int] = {}
    for pair in pairs:
        key, _, value = pair.partition("-")
        entities[key] = int(value) if key in _INDEX_KEYS and value.isdigit() else value
    return entities, suffix, extension


def _as_index(value) -> int | str:
    """An index entity value as a number: 1, "1" and "01" all give 1."""
    return int(value) if str(value).isdigit() else value


def find_padded(path: str | Path) -> list[Path]:
    """Return the files next to `path` that are the same file, whatever the zero-padding.

    A file matches when its suffix, extension and entities are those of
    `path`, with index entities (`run`, `echo`, `flip`, `inv`, `split`,
    `chunk`) compared as numbers: `sub-01_run-1_T1w.nii.gz` matches
    `sub-01_run-01_T1w.nii.gz`. Reads the folder itself, not the pybids
    index, so files written after the index was built are found.
    """
    path = Path(path)
    if not path.parent.is_dir():
        return []
    wanted = _filename_entities(path.name)
    return sorted(
        other for other in path.parent.iterdir()
        if other.name != path.name and _filename_entities(other.name) == wanted
    )
