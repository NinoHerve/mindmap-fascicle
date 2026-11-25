"""SimNIBS charm layout reader plugin.

SimNIBS is the umbrella software (head modeling + TMS/tES field
simulation); this module covers only `charm`, its segmentation/meshing
sub-pipeline (`m2m_sub-XX/segmentation/labeling.nii.gz` etc.) -- not
simulation outputs, and not the older pre-charm `headreco`/`mri2mesh`
segmentation pipelines, which have a different layout and would warrant
their own module if ever added.

Registers `seg` and `t1` (keys `charm-seg`/`charm-t1` by default), read
and written by file extension (nibabel `.nii.gz` out of the box) -- plain
images, not wrapped in any project-specific abstraction. `lut` is kept for
path building only and isn't registered.
"""

from functools import partial
from pathlib import Path
from typing import Any

from ..datatype import Datatype
from ..extension import register_extension
from ..naming import subject_label

_CHARM_MAP: dict[str, tuple[str, str]] = {
    "seg": ("segmentation", "labeling.nii.gz"),
    "lut": ("segmentation", "labeling_LUT.txt"),
    "t1": ("", "T1.nii.gz"),
}


def build_charm_path(root: str | Path, subject: str, datatype: str) -> Path:
    """Build a path into a SimNIBS charm subject directory.

    Parameters
    ----------
    root: str | Path
        Path to the charm output directory.
    subject: str
        Subject label (with or without a "sub-" prefix).
    datatype: str
        Key into the internal charm datatype map (e.g. "seg", "t1") --
        the bare, unprefixed name, not a registry key like "charm-seg".

    Raises
    ------
    ValueError
        If `datatype` is not a known charm datatype.
    """
    sub_id = subject_label(subject)
    sub_dir = Path(root) / f"sub-{sub_id}" / f"m2m_sub-{sub_id}"

    if datatype not in _CHARM_MAP:
        raise ValueError(f"Unknown charm datatype '{datatype}'. Available: {list(_CHARM_MAP)}")

    folder, filename = _CHARM_MAP[datatype]

    return sub_dir / folder / filename


DATATYPES: dict[str, Datatype] = {
    name: Datatype(partial(build_charm_path, datatype=name)) for name in ("seg", "t1")
}


def register(
    dm,
    root_key: str = "charm",
    root_path: str | Path | None = None,
    namespace: str | None = None,
    readers: dict[str, Any] | None = None,
    writers: dict[str, Any] | None = None,
) -> None:
    """Register SimNIBS charm datatypes on `dm`.

    Parameters
    ----------
    dm: fascicle.DatasetManager
    root_key: str
        Root key for the charm output directory.
    root_path: str | Path, optional
        Path to add as `root_key`. If omitted, `root_key` must already
        exist in `dm.roots`.
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
