"""The extension framework: how FreeSurfer, MNE, charm -- and yours -- plug in.

An extension is a module with a table of `Datatype`s keyed by bare name and
a `register` function that hands the table to `register_extension`:

    DATATYPES = {
        "t1": Datatype(partial(build_my_path, datatype="t1")),
        "seg": Datatype(partial(build_my_path, datatype="seg"), reader=read_seg),
    }

    def register(dm, root_key="mytool", root_path=None, namespace=None,
                 readers=None, writers=None):
        register_extension(dm, DATATYPES, root_key=root_key, root_path=root_path,
                           namespace=namespace, readers=readers, writers=writers)

The built-in extensions in `fascicle.extensions` are written exactly this way.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .datatype import Datatype


def register_extension(
    dm,
    datatypes: dict[str, Datatype],
    *,
    root_key: str,
    root_path: str | Path | None = None,
    namespace: str | None = None,
    readers: dict[str, Callable[[Any], Any]] | None = None,
    writers: dict[str, Callable[[Any, Any], None]] | None = None,
) -> None:
    """Register a table of datatypes on `dm` under one root.

    Parameters
    ----------
    dm: fascicle.DatasetManager
    datatypes: dict[str, Datatype]
        Bare name -> datatype. Each is registered as `f"{namespace}-{name}"`.
    root_key: str
        Root the datatypes resolve against.
    root_path: str | Path, optional
        Added as `root_key` if given; otherwise `root_key` must already
        exist in `dm.roots`.
    namespace: str, optional
        Key prefix. Defaults to `root_key`, so registering the same
        extension twice under different roots never collides.
    readers, writers: dict, optional
        Bare name -> function overriding that datatype's reader/writer.

    Raises
    ------
    KeyError
        If `root_path` is omitted and `root_key` isn't registered.
    ValueError
        If `readers`/`writers` name a datatype not in `datatypes`.
    """
    if root_path is not None:
        dm.add_root(root_key, root_path)
    elif root_key not in dm.roots:
        raise KeyError(
            f"Root '{root_key}' not found. Pass root_path=... to add it, "
            "or call dm.add_root() yourself first."
        )

    readers = readers or {}
    writers = writers or {}
    for label, overrides in (("readers", readers), ("writers", writers)):
        unknown = set(overrides) - set(datatypes)
        if unknown:
            raise ValueError(
                f"Unknown datatype(s) in {label}=: {sorted(unknown)}. "
                f"Available: {sorted(datatypes)}"
            )

    if namespace is None:
        namespace = root_key

    for name, dt in datatypes.items():
        changes: dict[str, Any] = {"root_key": root_key}
        if name in readers:
            changes["reader"] = readers[name]
        if name in writers:
            changes["writer"] = writers[name]
        dm.add(name, replace(dt, **changes), namespace=namespace)
