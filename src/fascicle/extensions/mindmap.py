"""mindmap extension: outputs of the mindmap tools.

For now, the BEM surfaces built by mindmap-bem from a head segmentation.
mindmap-bem writes them into an MNE subjects directory
(`mindmap-bem run --subjects-dir <root>`), under the names
`mne.make_bem_model` reads:

    <root>/sub-01/bem/inner_skull.surf     mm-iskull
    <root>/sub-01/bem/outer_skull.surf     mm-oskull
    <root>/sub-01/bem/outer_skin.surf      mm-skin

Keys are prefixed "mm" by default, as FreeSurfer's are "fs". They have no
underscores, closer to BIDS labels; the files keep MNE's names.

Read as `(coords, faces)`, coordinates in mm (tkRAS when mindmap-bem was
given the T1's volume info). Saving accepts `(coords, faces)` or
`(coords, faces, volume_info)`, keeping the volume info so MNE and
FreeSurfer read the surfaces in the right frame.

The layout is FreeSurfer's, so paths and surface I/O come from
`fascicle.extensions.freesurfer`. Surfaces from MNE's own BEM methods
(watershed, FLASH) are FreeSurfer-extension keys: `fs-iskull`, ...

More mindmap outputs will be added here.
"""

from functools import partial
from pathlib import Path
from typing import Any

from ..datatype import Datatype
from ..extension import register_extension
from .freesurfer import BEM_SURFACES, build_fs_path, read_surface, write_surface

DATATYPES: dict[str, Datatype] = {
    name: Datatype(partial(build_fs_path, datatype=name), read_surface, write_surface)
    for name in BEM_SURFACES
}


def register(
    dm,
    root_key: str = "mm",
    root_path: str | Path | None = None,
    namespace: str | None = None,
    readers: dict[str, Any] | None = None,
    writers: dict[str, Any] | None = None,
) -> None:
    """Register the mindmap datatypes on `dm`.

    Parameters
    ----------
    dm: fascicle.DatasetManager
    root_key: str
        Root key for the mindmap subjects directory.
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
