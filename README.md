# fascicle

Fascicle is a Python framework for reading, writing and managing neuroscience datasets. It is built around the BIDS standard and supports outputs from FreeSurfer, MNE-Python and SimNIBS.

The core of Fascicle reads and writes raw BIDS datasets. Extensions handle common neuroimaging derivatives, such as those from FreeSurfer, MNE-Python and SimNIBS CHARM. These extensions use the same small framework that you can use to write your own extensions, and you can replace any reader or writer with your own.

## Install

```bash
pip install mindmap-fascicle          # raw BIDS + MRI/surfaces/tractograms, FreeSurfer, SimNIBS charm
pip install "mindmap-fascicle[mne]"   # + raw MEG/EEG and MNE-Python source-analysis derivatives
```

The package installs as `mindmap-fascicle` but is imported as `fascicle`.

## Quickstart

```python
from fascicle import DatasetManager
from fascicle.extensions import freesurfer, mne

dm = DatasetManager("/data/my_study/rawdata")    # no config needed
dm.get_subjects()                                # ['01', '02', ...]

img = dm.load("01", "T1w", session="test")       # nibabel image
dm.save(img, "01", "T1w", session="retest")      # writes sub-01/ses-retest/anat/sub-01_ses-retest_T1w.nii.gz

dm.add_datatype("bold", "raw", dm.bids_namer("bold"))   # any other raw BIDS suffix
bold = dm.load("01", "bold", session="test", task="rest")

freesurfer.register(dm, root_path="/data/my_study/derivatives/freesurfer")
lh_coords, lh_faces = dm.load("01", "fs-pial", hemi="lh")

mne.register(dm, root_path="/data/my_study/derivatives/mne")
dm.save(cov, "01", "mne-cov", task="rest")       # sub-01/cov/sub-01_task-rest_cov.fif
info = dm.load("01", "meg-info", session="meg", task="rest", run=1)   # raw MEG, BIDS sidecars applied
dm.save(trans, "01", "mne-trans", session="meg")   # sub-01/ses-meg/trans/sub-01_ses-meg_trans.fif
scalp = dm.load("01", "fs-head-dense")[0]       # FreeSurfer subject's bem/sub-01-head-dense.fif
```

`dm.get_path(...)` takes the same arguments as `load` and returns the path without touching the file (except for raw modalities, below, which are looked up in the BIDS index). Index entities (`run`, `echo`, ...) may be zero-padded to any width in BIDS, so raw keys find the file whatever its padding: `run=1` finds `run-01` or `run-001`, `split=1` finds `split-01`. `save` writes a new file with the values as given (`run=1` → `run-1`, `run="01"` → `run-01`), and refuses to create `run-1` next to an existing `run-01`.

## Concepts

- **Subjects**: pass `"01"` or `"sub-01"`, both name the same files everywhere; `dm.get_subjects()` returns bare labels (`["01", ...]`).
- **Roots**: named folders (`dm.roots`). `"raw"` is the BIDS dataset; extensions add their own (`"fs"`, `"mne"`, ...) with `add_root`.
- **Datatypes**: every key in `dm.registry` (`"T1w"`, `"fs-pial"`, `"mne-cov"`) is a `Datatype` with
  - a **namer**, `namer(root, subject, **inputs) -> Path`, which only builds the path, so the same namer works for loading and saving. Inputs are BIDS entities (`session`, `task`, ...) or tool-specific ones (`hemi`);
  - a **reader**, `reader(path) -> data`;
  - a **writer**, `writer(data, path)`.

  A missing reader or writer means "use the one registered for the file extension". `dm.describe("fs-pial")` shows all of this for one key.
- **Raw modalities**: data a toolbox reads from the raw dataset itself, rather than from its own derivatives folder (MEG/EEG for MNE, or a modality BIDS doesn't define yet). `dm.add_raw(key, suffix, reader, extensions=...)` registers one read-only key that finds the file in the pybids index, so it works for any BIDS-shaped name (`sub-01[_ses-a]_task-x_<suffix>.<ext>`) whatever its extension, matches `run=1` to `run-01`, and returns folder recordings such as CTF `.ds`. The key needn't be the suffix: MNE's `meg-info` reads the `meg` files with a different reader. Extensions list these in a `RAW` table of `RawDatatype`s, registered unprefixed with `register_raw`. For files whose names aren't BIDS-shaped, use `add_datatype(key, "raw", your_namer)`.
- **Sidecars**: the small files describing a raw recording (`channels.tsv`, `meg.json`, `coordsystem.json`, ...), which BIDS lets sit higher up the tree (inheritance). `add_raw(key, suffix, sidecar_of="meg-file")` registers the `suffix` sidecar of a raw recording, found from the recording's entities by `fascicle.utils.find_sidecar`, e.g. a session-level `channels.tsv` from `run=1`. The MNE extension registers one key per sidecar (`meg-channels`, `meg-coordsystem`, ...), and its `meg`/`eeg`/`meg-info` keys apply the fields BIDS defines the same way everywhere: channel types and bad channels from `channels.tsv`, line frequency from `meg.json`. What changed is reported in `info["temp"]["bids_sidecars"]`; `meg-file`/`eeg-file`/`meg-info-file` return the file exactly as stored. Keys built with `bids_namer` (`T1w`, ...) follow inheritance too, for metadata files only (`.json`, `.tsv`, `.bval`, `.bvec`): `dm.load("01", "T1w", extension=".json")` falls back to `sub-01/sub-01_T1w.json` or a dataset-level `T1w.json`. `save` always writes the subject's own file, never a shared one.
- **File extensions**: `dm.ext_readers` / `dm.ext_writers` map extensions to default readers/writers: nibabel for images and tractograms (`.nii`, `.nii.gz`, `.mgh`, `.mgz`, `.gii`, `.trk`, `.tck`), the standard library for `.json` sidecars (read as dicts), numpy for `.bval`/`.bvec`, and pandas for `.tsv`/`.tsv.gz` tables (read as DataFrames, with BIDS's `n/a` as missing values).
- **References**: single files outside any subject (templates, atlases, lookup tables) loaded with `load_ref`. Each reference is read by file extension unless you give it its own reader, with `add_reference(key, path, reader=...)` or, for references from the config file, `set_ref_reader(key, reader)`. References have no writer: they're inputs, not outputs.
- **Config file** (optional): the declarative way to set up roots and references:

  ```yaml
  DIR_RAW: /data/my_study/rawdata          # -> "raw" root (a raw_root argument wins if both are given)
  DIR_TEMPLATES: /data/templates           # available as dm.config["DIR_TEMPLATES"] for add_root
  MRI_REF: /data/templates/MNI152.nii.gz   # any key containing "REF" becomes a reference: dm.load_ref("MRI_REF")
  ```

  `DatasetManager(config="config.yaml")`. Reading a config with `env=True` (the `read_config` default) also exports every top-level key as an environment variable.

`save` refuses to replace an existing file unless you pass `overwrite=True`, creates missing folders, and raises `TypeError` for read-only datatypes.

## Extensions

| extension | keys (default prefix) | extra dependency |
|---|---|---|
| `fascicle.extensions.freesurfer` | `fs-t1`, `fs-brain`, `fs-orig`, `fs-aseg`, `fs-white`/`pial`/`inflated`/`sphere`, `fs-area`/`curv`/`thickness`/`sulc`, `fs-cortex` (read-only), `fs-ras2tkras` (read-only), BEM surfaces `fs-iskull`/`oskull`/`skin` (`bem/inner_skull.surf`, `outer_skull.surf`, `outer_skin.surf`, as MNE's watershed/FLASH write them), MNE scalp surfaces `fs-head`/`fs-head-dense` (`bem/sub-XX-head[-dense].fif`, read-only, need `mne`) | none (`mne` for `fs-head*`) |
| `fascicle.extensions.charm` | `charm-seg`, `charm-t1` | none |
| `fascicle.extensions.mindmap` | BEM surfaces from mindmap-bem: `mm-iskull`, `mm-oskull`, `mm-skin` | none |
| `fascicle.extensions.mne` | raw (unprefixed, read-only): `meg`, `eeg`, `meg-info` (sidecars applied), `meg-file`, `eeg-file`, `meg-info-file` (as stored), sidecars `meg-`/`eeg-` + `channels`, `json`, `events`, `coordsystem`, plus `meg-headshape`, `eeg-electrodes`; derivatives: `mne-src`, `mne-fwd`, `mne-trans`, `mne-ave`, `mne-i-ave`, `mne-v-ave`, `mne-cov`, `mne-stc` | `mne` |

Every extension has the same `register(dm, root_key=..., root_path=None, namespace=None, readers=None, writers=None)`. Keys are prefixed with `namespace`, which defaults to `root_key`; raw keys (MNE's `meg`, ...) aren't, and registering an extension twice keeps one set of them. To compare two FreeSurfer runs, register twice: `freesurfer.register(dm, root_key="fs_run2", root_path=...)` gives `fs_run2-t1`, ... The same works for BEM methods kept in separate subjects directories: `freesurfer.register(dm, root_key="watershed", root_path=...)` gives `watershed-iskull`.

## Swapping a reader or writer

```python
import ants

def ants_reader(path):
    return ants.image_read(str(path))

# One datatype:
dm.set_reader("T1w", ants_reader)
freesurfer.register(dm, root_path=..., readers={"t1": ants_reader})   # at registration

# Every file of one type (raw BIDS, FreeSurfer, charm, ... alike):
dm.add_extension(".nii.gz", reader=ants_reader)
```

`set_writer`, `writers={...}` and `add_extension(..., writer=...)` work the same way. Mark a datatype read-only with `Datatype.READ_ONLY` as its writer.

## Writing your own extension

An extension is a table of `Datatype`s plus a `register` function. This is exactly how the built-in extensions are written. For BIDS-style names, build the namers with `derivative_namer`:

```python
# my_lab/resection.py
from fascicle import Datatype, derivative_namer, register_extension


def read_notes(path):
    return path.read_text()


def write_notes(text, path):
    path.write_text(text)


DATATYPES = {
    # nibabel by extension; the .json sidecar goes through the JSON reader/writer
    "mask": Datatype(derivative_namer("mask", datatype="anat", extensions=(".nii.gz", ".json"))),
    "notes": Datatype(
        derivative_namer("notes", datatype="anat", extensions=(".txt",), entities=("session", "desc")),
        read_notes,
        write_notes,
    ),
}


def register(dm, root_key="resec", root_path=None, namespace=None, readers=None, writers=None):
    register_extension(dm, DATATYPES, root_key=root_key, root_path=root_path,
                       namespace=namespace, readers=readers, writers=writers)
```

```python
resection.register(dm, root_path="/data/my_study/derivatives/resection")
dm.save(mask_img, "01", "resec-mask", session="postop", desc="manual")
# -> derivatives/resection/sub-01/ses-postop/anat/sub-01_ses-postop_desc-manual_mask.nii.gz
mask = dm.load("01", "resec-mask", session="postop", desc="manual")
```

### BIDS-style names with `derivative_namer`

`derivative_namer(suffix, datatype=None, extensions=(".nii.gz",), entities=None, extra_entities=(), fixed=None)` accepts any suffix, folder and extension:

- **`datatype`** is the folder under `sub-XX[/ses-YY]/`. Pass `None` to put files directly in it.
- **`extensions`**: the first one is the default; ask for another with `extension=`, e.g. a `.json` sidecar. `None` builds extensionless stems.
- **`entities`** restricts which BIDS entities the datatype takes. By default it takes all of them, using pybids' names (`session`, `acquisition`, `desc`, ...).
- **`extra_entities=("cut",)`** adds your own entities, written as `_cut-<value>` just before `desc`.
- **`fixed={"signal": "virtual"}`** gives an entity one value for this datatype: it's always written and callers can't pass it. Use it to give variants of a file their own keys while keeping one suffix. For example, the MNE extension's `mne-i-ave` / `mne-v-ave` are `ave` files with `signal-intracranial` / `signal-virtual`.

Entities are always written in the BIDS spec's order, whatever order you pass them in. Anything the namer wasn't declared with raises `ValueError` instead of being dropped. So do values the spec forbids, such as `desc="brain-mask"`, `run="one"` or `hemi="lh"` (BIDS uses `L`/`R`). `fascicle.bids_pattern(...)` shows the pybids pattern a namer fills.

`dm.bids_namer(suffix)` is the raw-BIDS counterpart: it knows the raw spec (T1w lives in `anat/`, bold in `func/`), but only accepts raw suffixes and entities. A derivative entity such as `desc` raises and points you to `derivative_namer`.

Namers don't have to be BIDS-style: any `namer(root, subject, **inputs) -> Path` works (see `fascicle.extensions.freesurfer`). For a one-off datatype, skip the module: `dm.add_root(...)` and `dm.add_datatype(key, root_key, namer, reader, writer)`. You can also subclass `DatasetManager` and register everything in `__init__`.

## Caveats

- **Not picklable.** `DatasetManager` wraps a `bids.BIDSLayout`, which can't be pickled. Don't pass a manager to `multiprocessing`/`joblib` workers. Resolve paths with `get_path` (or load data) in the parent process and pass those instead.
- **The BIDS index isn't refreshed after `save`.** Files you write are loadable by path straight away, but `get_subjects()` and `dm.layout` queries only see them in a new `DatasetManager`.
- **Extensions don't index their folders.** Namers build paths without looking at the disk, so they can write into folders that don't exist yet. For pybids queries on a derivatives folder, also call `dm.add_derivatives(key, path)` once it has a `dataset_description.json`. pybids doesn't parse custom entities (such as MNE's `signal`), so you can't filter on them in pybids queries; Fascicle's own `load`/`save` handle them fully.

## Example

`examples/tutorial.ipynb` downloads one subject of OpenNeuro's `ds000114`, loads its T1w and BOLD with nibabel, then switches T1w to ANTsPy. Run `pip install -e ".[examples]"` first.
