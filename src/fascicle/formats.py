"""Default readers and writers, one pair per file extension.

Imaging formats use nibabel, `.json` sidecars use the standard library,
`.bval`/`.bvec` diffusion tables use numpy, and `.tsv`/`.tsv.gz` tables use
pandas -- all core dependencies.

These defaults are part of Fascicle's core. When a datatype is registered 
without its own reader or writer, `DatasetManager.load_by_extension`
and `DatasetManager.save_by_extension` use these defaults.

Each `DatasetManager` calls `register(dm)` when it is created, so the
defaults are always available. To replace a default for the whole
manager, call `dm.add_extension(".nii.gz", reader=..., writer=...)`.
"""

import json
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd


def _read_image(path: str | Path) -> Any:
    """Read a NIfTI/MGH/MGZ/GIFTI image via nibabel."""
    return nib.load(path)


def _write_image(img: Any, path: str | Path) -> None:
    """Write a nibabel image (NIfTI/MGH/MGZ/GIFTI) to `path`."""
    nib.save(img, path)


def _read_streamlines(path: str | Path) -> Any:
    """Read a .trk/.tck tractogram via nibabel, returned unwrapped."""
    return nib.streamlines.load(path)


def _write_streamlines(tractogram: Any, path: str | Path) -> None:
    """Write a nibabel tractogram (or TractogramFile) to `path`."""
    nib.streamlines.save(tractogram, path)


def _read_json(path: str | Path) -> Any:
    """Read a JSON file (e.g. a BIDS sidecar) into Python objects."""
    with open(path) as f:
        return json.load(f)


def _write_json(data: Any, path: str | Path) -> None:
    """Write `data` as indented JSON, the usual layout of BIDS sidecars."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def _read_gradient_table(path: str | Path) -> np.ndarray:
    """Read a BIDS .bval (shape (N,)) or .bvec (shape (3, N)) file."""
    return np.loadtxt(path)


def _write_gradient_table(values: Any, path: str | Path) -> None:
    """Write a .bval (one row) or .bvec (three rows), space-separated as BIDS requires."""
    np.savetxt(path, np.atleast_2d(values), fmt="%.10g")


def _read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a BIDS .tsv/.tsv.gz table; BIDS's "n/a" becomes NaN."""
    return pd.read_csv(path, sep="\t", na_values="n/a", keep_default_na=False)


def _write_tsv(table: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as a BIDS table: tab-separated, no index, NaN as "n/a"."""
    table.to_csv(path, sep="\t", index=False, na_rep="n/a")


_FORMATS: dict[str, tuple[Callable[[Any], Any], Callable[[Any, Any], None]]] = {
    ".nii": (_read_image, _write_image),
    ".nii.gz": (_read_image, _write_image),
    ".mgh": (_read_image, _write_image),
    ".mgz": (_read_image, _write_image),
    ".gii": (_read_image, _write_image),
    ".trk": (_read_streamlines, _write_streamlines),
    ".tck": (_read_streamlines, _write_streamlines),
    ".json": (_read_json, _write_json),
    ".bval": (_read_gradient_table, _write_gradient_table),
    ".bvec": (_read_gradient_table, _write_gradient_table),
    ".tsv": (_read_tsv, _write_tsv),
    ".tsv.gz": (_read_tsv, _write_tsv),
}

DEFAULT_EXTENSIONS = tuple(_FORMATS)


def register(dm, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS) -> None:
    """Register the default readers and writers for `extensions` on `dm`.

    Raises
    ------
    ValueError
        If an extension has no default reader/writer.
    """
    for ext in extensions:
        if ext not in _FORMATS:
            raise ValueError(
                f"No default reader/writer for extension '{ext}'. Available: {list(_FORMATS)}"
            )
        reader, writer = _FORMATS[ext]
        dm.add_extension(ext, reader=reader, writer=writer)
