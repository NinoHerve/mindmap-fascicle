"""The core dataset manager: a datatype registry with built-in raw BIDS support.

`DatasetManager` resolves paths and loads and saves data in a BIDS dataset.
By default, it registers one root ("raw") and one datatype ("T1w"), plus
nibabel readers and writers for common imaging file extensions.

You add everything else through extensions or the public API. This
includes FreeSurfer, SimNIBS CHARM, MNE-Python and your own datatypes. For
extensions, see `fascicle.extensions` and `fascicle.register_extension`.
For the public API, see the `add_*` and `set_*` methods.

A `DatasetManager` cannot be pickled, because it wraps a `bids.BIDSLayout`,
which cannot be pickled either. So never pass an instance to a
multiprocessing or joblib worker.
"""

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Callable

from bids import BIDSLayout

from . import formats
from .config import read_config
from .datatype import Datatype
from .utils import build_bids_path, parse_extension


class DatasetManager:
    """Resolves paths, loads and saves data for a BIDS-organized dataset.

    Non-picklable: wraps `bids.BIDSLayout`, which cannot be pickled. Never
    pass an instance into a multiprocessing/joblib worker -- resolve paths
    via `get_path`/`load` in the parent process first.

    `save` does not re-index the BIDS layout: files written into the raw
    dataset are loadable by path, but `self.subjects` and `self.layout`
    queries won't see them until a new manager is created.
    """

    def __init__(
        self,
        raw_root: str | Path | None = None,
        config: str | Path | dict | None = None,
        validate: bool = False,
    ):
        if config is None:
            config = {}
        elif isinstance(config, (str, Path)):
            config = read_config(config)
        self.config = config

        self.roots: dict[str, str | Path] = {}
        self.refs = {k: v for k, v in self.config.items() if "REF" in k}
        self.ref_readers: dict[str, Callable[[Any], Any]] = {}
        self.registry: dict[str, Datatype] = {}
        self.ext_readers: dict[str, Callable[[Any], Any]] = {}
        self.ext_writers: dict[str, Callable[[Any, Any], None]] = {}

        root = raw_root if raw_root is not None else self.config.get("DIR_RAW")
        if root is None:
            raise KeyError(
                "No raw BIDS root given: pass raw_root=... or include "
                "'DIR_RAW' in the config."
            )
        self.add_root("raw", root)

        self.layout = BIDSLayout(self.roots["raw"], validate=validate)
        self.subjects = self.layout.get(return_type="id", target="subject")

        formats.register(self)
        self.add_datatype("T1w", "raw", self.bids_namer("T1w"))

    # -------------------------
    # Registration
    # -------------------------

    def add_root(self, root_key: str, path: str | Path) -> None:
        """Register a root folder under `root_key`."""
        self.roots[root_key] = path

    def add_reference(
        self, ref_key: str, path: str | Path, reader: Callable[[Any], Any] | None = None
    ) -> None:
        """Register a reference file under `ref_key`.

        `reader(path) -> data` is used by `load_ref` for this reference
        only; None reads it by file extension. Re-adding a key replaces
        both its path and its reader.
        """
        self.refs[ref_key] = path
        if reader is None:
            self.ref_readers.pop(ref_key, None)
        else:
            self.ref_readers[ref_key] = reader

    def set_ref_reader(self, ref_key: str, reader: Callable[[Any], Any] | None) -> None:
        """Replace the reader of reference `ref_key` (None = read by extension).

        Useful for references declared in the config file, which are
        registered without a reader.
        """
        self._ref_path(ref_key)
        if reader is None:
            self.ref_readers.pop(ref_key, None)
        else:
            self.ref_readers[ref_key] = reader

    def add_extension(
        self,
        extension: str,
        reader: Callable[[Any], Any] | None = None,
        writer: Callable[[Any, Any], None] | None = None,
    ) -> None:
        """Set the reader and/or writer used for files ending in `extension`.

        Applies to every datatype registered without its own reader/writer
        (and to `load_ref`), e.g. `dm.add_extension(".nii.gz", ants.image_read)`
        switches every NIfTI read to ANTs.
        """
        if reader is not None:
            self.ext_readers[extension] = reader
        if writer is not None:
            self.ext_writers[extension] = writer

    def add(self, key: str, datatype: Datatype, *, namespace: str | None = None) -> str:
        """Register a ready-made `Datatype` under `key` (or `f"{namespace}-{key}"`).

        Returns the registry key used.
        """
        full_key = f"{namespace}-{key}" if namespace is not None else key
        self.registry[full_key] = datatype
        return full_key

    def add_datatype(
        self,
        key: str,
        root_key: str | None,
        namer: Callable[..., Path],
        reader: Callable[[Any], Any] | None = None,
        writer: Callable[[Any, Any], None] | None = None,
        *,
        inputs: tuple[str, ...] = (),
        exists: Callable[[Path], bool] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Register a datatype for path building, loading and saving.

        Parameters
        ----------
        key: str
            Registry key (prefixed with `namespace` if given).
        root_key: str | None
            Registered root the namer resolves against, or None if the
            namer finds its own way (it then receives `root=None`).
        namer: Callable
            `namer(root, subject, **inputs) -> Path`. Must not check that
            the file exists.
        reader: Callable, optional
            `reader(path) -> data`. None dispatches on file extension.
        writer: Callable, optional
            `writer(data, path)`. None dispatches on file extension.
        inputs: tuple[str, ...]
            Names of the extra inputs `namer` expects (documentation).
        exists: Callable, optional
            Replaces `Path.exists` for namers returning a multi-file stem.
        namespace: str, optional
            Registers under `f"{namespace}-{key}"`, so extensions with
            overlapping vocabularies (FreeSurfer's and charm's "t1") don't
            collide.

        Returns the registry key used.
        """
        dt = Datatype(namer, reader, writer, tuple(inputs), exists, root_key)
        return self.add(key, dt, namespace=namespace)

    def set_reader(self, key: str, reader: Callable[[Any], Any] | None) -> None:
        """Replace the reader of datatype `key` (None = dispatch on extension)."""
        self.registry[key] = replace(self._entry(key), reader=reader)

    def set_writer(self, key: str, writer: Callable[[Any, Any], None] | None) -> None:
        """Replace the writer of datatype `key` (None = dispatch on extension)."""
        self.registry[key] = replace(self._entry(key), writer=writer)

    def add_derivatives(
        self, root_key: str, path: str | Path, config: str | Path | dict | None = None
    ) -> None:
        """Register a BIDS-derivatives root, indexed into `self.layout`.

        Wraps `add_root(root_key, path)` and `self.layout.add_derivatives`.
        The folder must already exist with a `dataset_description.json`.
        """
        self.add_root(root_key, path)
        self.layout.add_derivatives(path, config=config)

    def bids_namer(self, suffix: str) -> Callable[..., Path]:
        """Return a namer building raw-BIDS paths for `suffix` from entities.

        e.g. `dm.add_datatype("bold", "raw", dm.bids_namer("bold"))`, then
        `dm.load("01", "bold", session="test", task="rest")`.
        """

        def namer(root: Any, subject: str, **entities) -> Path:
            return Path(build_bids_path(self.layout, subject, suffix, **entities))

        namer.__name__ = f"bids_namer({suffix!r})"
        return namer

    # -------------------------
    # Access
    # -------------------------

    def get_path(self, subject: str, key: str, **inputs) -> Path:
        """Build the path of datatype `key` for `subject`. Doesn't check it exists."""
        dt = self._entry(key)
        root = self.roots[dt.root_key] if dt.root_key is not None else None
        return Path(dt.namer(root, subject, **inputs))

    def load(self, subject: str, key: str, **inputs) -> Any:
        """Build the path of datatype `key` for `subject` and read it."""
        if subject not in self.subjects:
            raise ValueError(f"Subject sub-{subject} does not exist in dataset.")

        dt = self._entry(key)
        path = self.get_path(subject, key, **inputs)

        if not self._exists(dt, path):
            raise FileNotFoundError(f"No file for '{key}' (sub-{subject}) at: {path}")

        reader = dt.reader or self.load_by_extension
        return reader(path)

    def save(self, data: Any, subject: str, key: str, *, overwrite: bool = False, **inputs) -> Path:
        """Write `data` as datatype `key` for `subject`; returns the path written.

        Raises
        ------
        TypeError
            If `key` is read-only (no writer, and none for its extension).
        FileExistsError
            If the file exists and `overwrite` is False.
        """
        dt = self._entry(key)
        path = self.get_path(subject, key, **inputs)

        writer = dt.writer
        if writer is Datatype.READ_ONLY:
            raise TypeError(f"Datatype '{key}' is read-only. Use set_writer() to add a writer.")
        if writer is None:
            if parse_extension(path) not in self.ext_writers:
                raise TypeError(
                    f"Datatype '{key}' is read-only: it has no writer and no writer "
                    f"is registered for '{parse_extension(path)}'. Use set_writer() "
                    "or add_extension() to add one."
                )
            writer = self.save_by_extension

        if not overwrite and self._exists(dt, path):
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")

        path.parent.mkdir(parents=True, exist_ok=True)
        writer(data, path)
        return path

    def load_ref(self, ref_key: str) -> Any:
        """Read reference `ref_key` with its own reader, or by file extension."""
        path = self._ref_path(ref_key)
        reader = self.ref_readers.get(ref_key, self.load_by_extension)
        return reader(path)

    def load_by_extension(self, path: str | Path) -> Any:
        """Read `path` with the reader registered for its extension."""
        extension = parse_extension(path)

        if extension not in self.ext_readers:
            raise ValueError(
                f"No reader registered for extension '{extension}'. "
                "Use add_extension() to register one."
            )

        return self.ext_readers[extension](path)

    def save_by_extension(self, data: Any, path: str | Path) -> None:
        """Write `data` to `path` with the writer registered for its extension."""
        extension = parse_extension(path)

        if extension not in self.ext_writers:
            raise ValueError(
                f"No writer registered for extension '{extension}'. "
                "Use add_extension() to register one."
            )

        self.ext_writers[extension](data, path)

    def get_subjects(self) -> list[str]:
        """Return the dataset's subject list."""
        return self.subjects

    def describe(self, key: str) -> dict[str, Any]:
        """Summarize datatype `key`: root, namer, reader, writer and inputs."""
        dt = self._entry(key)

        def name(func, fallback):
            if func is None:
                return fallback
            if isinstance(func, partial):
                bound = ", ".join(f"{k}={v!r}" for k, v in func.keywords.items())
                return f"{name(func.func, fallback)}({bound})"
            return getattr(func, "__name__", repr(func))

        return {
            "key": key,
            "root_key": dt.root_key,
            "root": self.roots.get(dt.root_key) if dt.root_key is not None else None,
            "namer": name(dt.namer, None),
            "reader": name(dt.reader, "by extension"),
            "writer": name(dt.writer, "by extension"),
            "inputs": dt.inputs,
        }

    # -------------------------
    # Internal
    # -------------------------

    def _entry(self, key: str) -> Datatype:
        if key not in self.registry:
            raise KeyError(
                f"No registry entry for datatype '{key}'. "
                f"Registered datatypes: {sorted(self.registry)}. "
                "Use add_datatype() to register a custom datatype."
            )
        return self.registry[key]

    def _ref_path(self, ref_key: str) -> str | Path:
        if ref_key not in self.refs:
            raise KeyError(
                f"No reference registered for '{ref_key}'. "
                f"Registered references: {sorted(self.refs)}. "
                "Use add_reference() to register one."
            )
        return self.refs[ref_key]

    @staticmethod
    def _exists(dt: Datatype, path: Path) -> bool:
        return dt.exists(path) if dt.exists is not None else path.exists()
