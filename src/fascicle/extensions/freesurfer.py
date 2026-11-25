"""FreeSurfer recon-all extension.

Covers a recon-all subjects directory (`<root>/sub-XX/{mri,surf,label,bem}`).
Registered keys are prefixed with the namespace (default: the root key,
"fs"): `fs-t1`, `fs-pial`, ... Surface, morphometry and label datatypes
are per hemisphere: `dm.load("01", "fs-pial", hemi="lh")`.

| name                           | reader                   | writer              |
|--------------------------------|--------------------------|---------------------|
| t1, brain, orig, aseg          | by extension (.mgz)      | by extension        |
| white, pial, inflated, sphere  | read_geometry            | write_geometry      |
| area, curv, thickness, sulc    | read_morph_data          | write_morph_data    |
| cortex                         | read_label               | none (read-only)    |
| ras2tkras                      | from the T1.mgz header   | none (derived)      |
| iskull, oskull, skin           | read_surface             | write_surface       |
| (bem/*.surf)                   |                          |                     |

Swap any reader/writer with `register(dm, readers={"t1": ants.image_read})`
or `dm.set_reader("fs-t1", ...)`.

The BEM surfaces are those MNE writes into a subject's `bem/` folder
(`mne watershed_bem --copy`, `mne flash_bem --copy`), under the names
`mne.make_bem_model` reads: `iskull` is `inner_skull.surf`, `oskull`
`outer_skull.surf` and `skin` `outer_skin.surf` (keys without underscores,
closer to BIDS labels; the files keep MNE's names). To compare BEM methods
kept in separate subjects directories, register once per folder:

    freesurfer.register(dm, root_key="watershed", root_path=".../bem-watershed")
    dm.load("01", "watershed-iskull")        # (coords, faces), tkRAS mm

`vol-info` and mindmap's own BEM loaders stay mindmap-side (see PLAN.md);
mindmap-bem's surfaces are in `fascicle.extensions.mindmap` (`mm-iskull`, ...).
"""

from functools import partial
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from ..datatype import Datatype
from ..extension import register_extension
from ..naming import subject_label

_FS_MAP: dict[str, tuple[str, str]] = {
    "t1": ("mri", "T1.mgz"),
    "brain": ("mri", "brain.mgz"),
    "orig": ("mri", "orig.mgz"),
    "aseg": ("mri", "aseg.mgz"),
    "white": ("surf", "{hemi}.white"),
    "pial": ("surf", "{hemi}.pial"),
    "inflated": ("surf", "{hemi}.inflated"),
    "sphere": ("surf", "{hemi}.sphere"),
    "area": ("surf", "{hemi}.area"),
    "curv": ("surf", "{hemi}.curv"),
    "thickness": ("surf", "{hemi}.thickness"),
    "sulc": ("surf", "{hemi}.sulc"),
    "cortex": ("label", "{hemi}.cortex.label"),
    "iskull": ("bem", "inner_skull.surf"),
    "oskull": ("bem", "outer_skull.surf"),
    "skin": ("bem", "outer_skin.surf"),
}

# BEM surface keys; files keep MNE's names (see _FS_MAP)
BEM_SURFACES = ("iskull", "oskull", "skin")


def build_fs_path(
    root: str | Path, subject: str, datatype: str, hemi: str | None = None
) -> Path:
    """Build a path into a FreeSurfer recon-all subject directory.

    Parameters
    ----------
    root: str | Path
        Path to the FreeSurfer subjects directory.
    subject: str
        Subject label (with or without a "sub-" prefix).
    datatype: str
        Key into the internal FreeSurfer datatype map (e.g. "white", "t1")
        -- the bare, unprefixed name, not a registry key like "fs-t1".
    hemi: str, optional
        "lh" or "rh". Required for hemisphere-specific datatypes (surfaces,
        morphometry, labels), ignored otherwise.

    Raises
    ------
    ValueError
        If `datatype` is not a known FreeSurfer datatype, or is
        hemisphere-specific and `hemi` isn't "lh"/"rh".
    """
    sub_id = subject_label(subject)
    sub_dir = Path(root) / f"sub-{sub_id}"

    if datatype not in _FS_MAP:
        raise ValueError(f"Unknown FS datatype '{datatype}'. Available: {list(_FS_MAP)}")

    folder, file_tmpl = _FS_MAP[datatype]
    if "{hemi}" in file_tmpl and hemi not in ("lh", "rh"):
        raise ValueError(f"FS datatype '{datatype}' needs hemi='lh' or hemi='rh', got {hemi!r}.")
    filename = file_tmpl.format(hemi=hemi)

    return sub_dir / folder / filename


def read_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a FreeSurfer surface as `(coords, faces)` (coords in tkRAS mm)."""
    return nib.freesurfer.io.read_geometry(path)


def write_surface(surface: tuple, path: str | Path) -> None:
    """Write a FreeSurfer surface from `(coords, faces)` or `(coords, faces, volume_info)`.

    `volume_info` (as `read_geometry(..., read_metadata=True)` returns it)
    records the volume whose tkRAS the coordinates are in; MNE and
    FreeSurfer read it back.
    """
    coords, faces, *volume_info = surface
    nib.freesurfer.io.write_geometry(
        path, coords, faces, volume_info=volume_info[0] if volume_info else None
    )


def _write_morph_data(values: np.ndarray, path: str | Path) -> None:
    """Write per-vertex morphometry (nibabel's own writer takes path first)."""
    nib.freesurfer.io.write_morph_data(path, values)


def _read_ras2tkras(path_to_t1: str | Path) -> np.ndarray:
    """Compute the RAS-to-TkReg-RAS transform from a t1 (T1.mgz) header."""
    mri = nib.load(path_to_t1)

    vox2ras = mri.header.get_vox2ras()
    vox2ras_tkr = mri.header.get_vox2ras_tkr()

    return vox2ras_tkr @ np.linalg.inv(vox2ras)


def _namer(datatype: str):
    return partial(build_fs_path, datatype=datatype)


_HEMI = ("hemi",)

DATATYPES: dict[str, Datatype] = {
    **{name: Datatype(_namer(name)) for name in ("t1", "brain", "orig", "aseg")},
    **{
        name: Datatype(
            _namer(name), nib.freesurfer.io.read_geometry, write_surface, inputs=_HEMI
        )
        for name in ("white", "pial", "inflated", "sphere")
    },
    **{
        name: Datatype(
            _namer(name),
            nib.freesurfer.io.read_morph_data,
            _write_morph_data,
            inputs=_HEMI,
        )
        for name in ("area", "curv", "thickness", "sulc")
    },
    "cortex": Datatype(_namer("cortex"), nib.freesurfer.io.read_label, Datatype.READ_ONLY, inputs=_HEMI),
    # Derived from T1.mgz's header: its path is T1.mgz, so it must be
    # explicitly read-only or saving would overwrite the T1 via the .mgz writer.
    "ras2tkras": Datatype(_namer("t1"), _read_ras2tkras, Datatype.READ_ONLY),
    **{name: Datatype(_namer(name), read_surface, write_surface) for name in BEM_SURFACES},
}


def register(
    dm,
    root_key: str = "fs",
    root_path: str | Path | None = None,
    namespace: str | None = None,
    readers: dict[str, Any] | None = None,
    writers: dict[str, Any] | None = None,
) -> None:
    """Register FreeSurfer recon-all datatypes on `dm`.

    Parameters
    ----------
    dm: fascicle.DatasetManager
    root_key: str
        Root key for the FreeSurfer subjects directory.
    root_path: str | Path, optional
        Path to add as `root_key`. If omitted, `root_key` must already
        exist in `dm.roots`.
    namespace: str, optional
        Key prefix; defaults to `root_key`, so registering twice under
        different root keys (e.g. to compare two recon-all runs) never
        collides.
    readers, writers: dict, optional
        Bare datatype name -> replacement function, e.g.
        `readers={"t1": ants.image_read}`.

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
