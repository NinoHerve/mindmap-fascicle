"""The `Datatype` record: how Fascicle names, reads and writes one kind of file.

A datatype combines three parts: a *namer*, which builds the file's path,
a *reader* and a *writer*.

Namers are pure functions. A namer builds a path from a root, a subject and
any extra inputs (BIDS entities, a FreeSurfer hemisphere, ...). It never
checks whether the file exists, so `load` and `save` can share the same namer.

Readers and writers each handle exactly one file: `reader(path)` loads it,
and `writer(data, path)` saves it. Because this matches the signature of
plain library functions such as `nibabel.load` or `ants.image_read`, you can
pass those functions directly. If the reader or writer is `None`, the
manager picks one from its per-extension registries based on the file
extension.

To forbid saving a datatype, pass `writer=Datatype.READ_ONLY`. Use this when the
datatype's path ends in an extension that has a registered writer, but the
file does not hold the datatype's data. For example, a transform derived
from a T1 header points to the T1 file. Without it, `save` would
write the transform through the NIfTI writer and overwrite the T1 image.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Datatype:
    """One registered kind of file.

    Attributes
    ----------
    namer: Callable
        `namer(root, subject, **inputs) -> Path`. Pure: must not check
        that the file exists. Bind datatype-specific arguments with
        `functools.partial` (e.g. `partial(build_fs_path, datatype="pial")`).
    reader: Callable, optional
        `reader(path) -> data`. None dispatches on the file extension.
    writer: Callable, optional
        `writer(data, path) -> None`. None dispatches on the file
        extension; `Datatype.READ_ONLY` forbids saving. A datatype with no writer
        and no writer for its extension is also read-only.
    inputs: tuple[str, ...]
        Names of the extra inputs the namer expects (e.g. `("hemi",)`).
        Documentation only, surfaced by `DatasetManager.describe`.
    exists: Callable, optional
        `exists(path) -> bool`, used by `load`/`save` instead of
        `Path.exists` -- for namers that return a stem standing for
        several files (e.g. MNE's `-lh.stc`/`-rh.stc` pairs).
    root_key: str, optional
        Which registered root the namer resolves against, or None if the
        namer finds its own way (it then receives `root=None`).
    """

    namer: Callable[..., Path]
    reader: Callable[[Path], Any] | None = None
    writer: Callable[[Any, Path], None] | None = None
    inputs: tuple[str, ...] = ()
    exists: Callable[[Path], bool] | None = None
    root_key: str | None = None

    # Not a field (no annotation): a writer sentinel, used as
    # `writer=Datatype.READ_ONLY`.
    @staticmethod
    def READ_ONLY(data: Any, path: Path) -> None:
        """Writer marking a datatype as read-only; `DatasetManager.save` refuses it."""
        raise TypeError(f"Datatype at {path} is read-only.")


@dataclass(frozen=True)
class RawDatatype:
    """A raw modality found through the dataset's BIDS index.

    Declares what `DatasetManager.add_raw` registers: a pybids query
    (`suffix`, `extensions`, `filters`) and a reader. Extensions list
    these in a `RAW` table and hand it to `fascicle.register_raw`.

    The registry key is chosen separately, so one suffix can back several
    keys with different readers (e.g. MNE's `meg` and `meg-info`).

    Attributes
    ----------
    suffix: str
        BIDS suffix to look up, e.g. "meg". Any suffix pybids indexes
        works, including ones the BIDS spec doesn't define.
    reader: Callable, optional
        `reader(path) -> data`. None dispatches on the file extension.
    extensions: tuple[str, ...], optional
        Extensions that hold the recording, e.g. (".vhdr", ".edf"), so
        companion files (".eeg", ".vmrk", ".json") don't match. None
        accepts any extension.
    filters: dict, optional
        Extra pybids filters always applied, e.g. `{"datatype": "eeg"}`.
    sidecar_of: str, optional
        Key of a raw recording: this is then that recording's `suffix`
        sidecar, found by BIDS inheritance (e.g. `channels.tsv` of `meg`).
    """

    suffix: str
    reader: Callable[[Path], Any] | None = None
    extensions: tuple[str, ...] | None = None
    filters: dict[str, Any] | None = None
    sidecar_of: str | None = None
