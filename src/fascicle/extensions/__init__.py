"""Built-in extensions for common neuroimaging derivatives.

Each submodule (`freesurfer`, `charm`, `mne`, `mindmap`) is a table of `Datatype`s plus a
`register(dm, root_key=..., root_path=None, namespace=None, readers=None,
writers=None)` function built on `fascicle.register_extension` -- the same
framework third-party extensions use.
"""
