"""MNE-Python extension: source-analysis derivatives with BIDS-style names.

Registers `src`, `fwd`, `ave`, `i-ave`, `v-ave`, `cov` and `stc` (keys
`mne-src`, `mne-cov`, ... by default) under one derivatives root, named with
`fascicle.derivative_namer`, e.g.

    <root>/sub-01/cov/sub-01_task-rest_cov.fif
    <root>/sub-01/ave/sub-01_task-rest_ave.fif                       (ave)
    <root>/sub-01/ave/sub-01_task-rest_signal-intracranial_ave.fif   (i-ave)
    <root>/sub-01/ave/sub-01_task-rest_signal-virtual_ave.fif        (v-ave)
    <root>/sub-01/stc/sub-01_task-rest_stc        (stem: -lh.stc/-rh.stc)

`i-ave` is the evoked response measured at intracranial contacts; `v-ave`
is the "virtual" one, estimated by projecting a source reconstruction onto
those contact locations. Both keep MNE's `_ave.fif` naming, told apart from
the scalp `ave` by a custom `signal` entity fixed per key (you never pass it).

Extra inputs are BIDS entities: `session`, `task`, `run`, `desc` (`src`/`fwd`
take only `session`/`desc`). Unknown entities raise instead of being dropped.

Paths are built without looking at the disk, so the folder doesn't need to
exist yet (writing into a fresh derivatives folder works). For pybids
queries on top, also call `dm.add_derivatives(root_key, root_path)` once
the folder has a `dataset_description.json`.

`mne` is an optional dependency (`pip install fascicle[mne]`): it's only
imported when a file is actually read or written.
"""

from pathlib import Path
from typing import Any

from ..datatype import Datatype
from ..extension import register_extension
from ..naming import derivative_namer

_FILE_ENTITIES = ("session", "task", "run", "desc")
_ANAT_ENTITIES = ("session", "desc")


def _mne():
    try:
        import mne
    except ImportError as err:  # pragma: no cover - exercised only without mne
        raise ImportError(
            "Reading/writing MNE derivatives needs mne: pip install fascicle[mne]"
        ) from err
    return mne


def _read_src(path):
    return _mne().read_source_spaces(path, verbose="warning")


def _write_src(src, path):
    _mne().write_source_spaces(path, src, overwrite=True, verbose="warning")


def _read_fwd(path):
    return _mne().read_forward_solution(path, verbose="warning")


def _write_fwd(fwd, path):
    _mne().write_forward_solution(path, fwd, overwrite=True, verbose="warning")


def _read_evoked(path):
    return _mne().read_evokeds(path, verbose="warning")[0]


def _write_evoked(evoked, path):
    _mne().write_evokeds(path, evoked, overwrite=True, verbose="warning")


def _read_cov(path):
    cov = _mne().read_cov(path, verbose="warning")
    cov["data"] = cov["data"].astype("float")
    return cov


def _write_cov(cov, path):
    cov.save(path, overwrite=True, verbose="warning")


def _read_stc(stem):
    return _mne().read_source_estimate(stem)


def _write_stc(stc, stem):
    stc.save(stem, overwrite=True, verbose="warning")


def _stc_exists(stem: Path) -> bool:
    """A surface stc stem stands for a -lh.stc/-rh.stc pair (or one -stc.h5 / -vl.stc)."""
    pair = all(Path(f"{stem}-{hemi}.stc").exists() for hemi in ("lh", "rh"))
    return pair or any(Path(f"{stem}{ext}").exists() for ext in ("-stc.h5", "-vl.stc"))


def _dt(suffix, reader, writer, entities=_FILE_ENTITIES, extensions=(".fif",), exists=None,
        signal=None):
    fixed = {"signal": signal} if signal else None
    extra = ("signal",) if signal else ()
    namer = derivative_namer(suffix, suffix, extensions, entities, extra, fixed)
    return Datatype(namer, reader, writer, entities, exists)


DATATYPES: dict[str, Datatype] = {
    "src": _dt("src", _read_src, _write_src, _ANAT_ENTITIES),
    "fwd": _dt("fwd", _read_fwd, _write_fwd, _ANAT_ENTITIES),
    "ave": _dt("ave", _read_evoked, _write_evoked),
    "i-ave": _dt("ave", _read_evoked, _write_evoked, signal="intracranial"),
    "v-ave": _dt("ave", _read_evoked, _write_evoked, signal="virtual"),
    "cov": _dt("cov", _read_cov, _write_cov),
    "stc": _dt("stc", _read_stc, _write_stc, extensions=None, exists=_stc_exists),
}


def register(
    dm,
    root_key: str = "mne",
    root_path: str | Path | None = None,
    namespace: str | None = None,
    readers: dict[str, Any] | None = None,
    writers: dict[str, Any] | None = None,
) -> None:
    """Register MNE source-analysis datatypes on `dm`.

    Parameters
    ----------
    dm: fascicle.DatasetManager
    root_key: str
        Root key for the MNE derivatives folder.
    root_path: str | Path, optional
        Path to add as `root_key` (needn't exist yet). If omitted,
        `root_key` must already exist in `dm.roots`.
    namespace: str, optional
        Key prefix; defaults to `root_key`.
    readers, writers: dict, optional
        Bare datatype name -> replacement function.

    Raises
    ------
    KeyError
        If `root_path` is omitted and `root_key` isn't already registered.
    """
    register_extension(
        dm,
        DATATYPES,
        root_key=root_key,
        root_path=root_path,
        namespace=namespace,
        readers=readers,
        writers=writers,
    )
