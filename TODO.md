# TODO

## Requests from the mindmap-bem manuscript (2026-11-24)

From `Manuscripts/mindmap-bem`, which reads OpenNeuro ds000117 (tag 1.1.0): anatomy plus
one MEG/EEG run per subject, for EEG/MEG forward models. Tested with mindmap-fascicle 0.1.0
on `sub-01`. Until these are in the package, the manuscript registers what it needs in
its own `processing/common.py:dataset()`.

Files involved (BIDS-MEG, Elekta VectorView):

```
sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv      # session level (inheritance)
sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json          # session level (inheritance)
sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json
sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos
sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif
```

- [x] **Raw MEG/EEG datatypes with an MNE reader.** Done: `mne.register` adds raw `meg`, `eeg`
  and `meg-info` (unprefixed, read-only), found through the BIDS index by the new core
  `dm.add_raw` / `register_raw`. `meg-info` uses `mne.io.read_info` for `.fif`, the raw header
  otherwise. Not covered: BTi/4D (no extension, not read by `mne.io.read_raw`).
  Original request: Only `T1w` is built in, and the MNE
  extension (`extensions/mne.py`) covers derivatives only (`src`, `fwd`, `ave`, `cov`,
  `stc`). `dm.add_datatype("meg", "raw", dm.bids_namer("meg"))` builds the right path, but
  `dm.load(...)` fails: `No reader registered for extension '.fif'`.
  - Suggested: raw datatypes `meg` and `eeg` in `fascicle.extensions.mne` (lazy `mne`
    import, like the other MNE readers), read with `mne.io.read_raw` (any MNE-supported
    format: `.fif`, `.ds`, `.con`, `.edf`, `.vhdr`, ...), `preload=False`.
  - Also useful: a way to read only the measurement info (`mne.io.read_info`). It's all
    a forward model needs (sensors, digitisation, `dev_head_t`), and much faster than
    opening an 870 MB recording. E.g. a separate datatype `meg-info`, or a `reader`
    option.
  - Raw recordings should be read-only (`writer=Datatype.READ_ONLY`).
- [x] **Readers for the other BIDS-MEG/EEG sidecars.** Done: one key per sidecar
  (`meg-channels`, `meg-json`, `meg-events`, `meg-coordsystem`, `meg-headshape`, and the `eeg-`
  ones plus `eeg-electrodes`). `meg`/`eeg`/`meg-info` now apply channel types and bads from
  `channels.tsv` and `PowerLineFrequency` from the json, report it in
  `info["temp"]["bids_sidecars"]`; `meg-file`/`eeg-file`/`meg-info-file` are the plain files.
  In ds000117 this fixes `EEG061`-`064` (HEOG/VEOG/ECG/MISC, typed EEG in the `.fif`).
  For FIF, `headshape.pos` matches the file's digitisation exactly, so it isn't applied.
  Still open: `.hsp`/`.elp` headshapes; applying `electrodes.tsv` montages or events.
  Original request:
  - `_headshape.pos` (and `.hsp`/`.elp`): no reader for `.pos`. In ds000117 it's plain
    text, `label x y z` in mm (`EEG1`..`EEG70`, `HP1`..`HP54`). A reader could return
    `{label: xyz}` or an `mne.channels.DigMontage`. The units and frame are in
    `coordsystem.json`.
  - `_electrodes.tsv` (EEG): `.tsv` already loads through the default reader.
    Registering the suffix as a datatype is probably enough, but check.
  - `_coordsystem.json`: already loads as a dict through the default `.json` reader.
    Registering the suffix (`coordsystem`) would make it discoverable through `describe`.
- [x] **BIDS inheritance: files above the datatype folder aren't found.** Done for raw keys:
  `fascicle.utils.find_sidecar` and `add_raw(..., sidecar_of=...)`. Still open for
  `bids_namer` keys such as `T1w` sidecars.
  - `channels.tsv` and `meg.json` sit at the session level (`ses-meg/`), which the BIDS
    inheritance principle allows. `bids_namer("channels")` builds
    `ses-meg/meg/..._channels.tsv`, which doesn't exist, so the load raises
    `FileNotFoundError`.
  - pybids indexes the file correctly: `dm.layout.get(subject="01", session="meg",
    suffix="channels")` returns it.
  - Suggested: for raw reads, fall back to the layout (`layout.get(...)` /
    `layout.get_nearest(...)`) when the built path doesn't exist, or resolve sidecars
    with pybids' inheritance (`get_metadata`-style lookup).
- [x] **The `run` entity isn't zero-padded to match the dataset.** Done: `bids_namer` keys
  (`T1w`, ...) find the same file under any zero-padding of any index entity
  (`fascicle.utils.find_padded`), `save` writes values as given and won't duplicate a file
  under another padding. `add_raw` keys match `run` through pybids. Still open: pybids
  compares only `run` as a number (`echo`/`flip`/`inv`/`chunk` are strings, `split` isn't
  parsed), so `add_raw` lookups with e.g. MEG `split-01` would need the same treatment.
  - `get_path(..., run=1)` builds `..._run-1_meg.fif`, but ds000117 names it `run-01`.
    The path is wrong, and the error only says "No file for 'meg'".
  - `run="01"` works, so it's a usability issue, not wrong data: BIDS allows both forms,
    and pybids `build_path` uses the value as given.
  - Suggested: when the built path doesn't exist, look up the run in the layout, which
    stores runs as integers, and use the indexed file. Otherwise mention the padding in
    the error message.

## Requests from the mindmap-bem manuscript (2026-11-25, release 0.2.0)

- [x] **`sub-` prefix handled differently by core and extension keys.** (fixed in 0.3.0) The FreeSurfer, charm
  and raw (`add_raw`) namers strip a `sub-` prefix, so `"sub-01"` and `"01"` both work. The core
  `bids_namer` keys and `load` don't:
  - `dm.get_path("sub-01", "T1w", session="mri", acquisition="mprage")` builds
    `sub-sub-01/ses-mri/anat/sub-sub-01_ses-mri_acq-mprage_T1w.nii.gz`;
  - `dm.load("sub-01", "meg-coordsystem", session="meg")` raises
    `Subject sub-sub-01 does not exist in dataset.`, even though `get_path` finds the file.
  - Suggested: strip `sub-` once, at the top of `get_path`/`load`/`save`, so every key
    accepts both forms. Alternatively, reject `sub-` everywhere with a clear error.
  - Manuscript workaround: it passes bare labels (`common.label()`).

## Requests (2026-11-25)

- [x] **A subject given as `"sub-01"` or `"01"` names the same files.** Extension namers
  already stripped `sub-`; the core didn't (`get_path("sub-01", "T1w")` built
  `sub-sub-01/...`, `load` raised `Subject sub-sub-01 does not exist`). Done: `get_path`,
  `load`, `save` and `bids_namer` strip a leading `sub-` (`fascicle.naming.subject_label`).
- [x] **BEM surfaces, one key per surface.** Done:
  - FreeSurfer extension: `fs-iskull`, `fs-oskull`, `fs-skin` (files
    `<root>/sub-XX/bem/inner_skull.surf`, `outer_skull.surf`, `outer_skin.surf`, as MNE's
    watershed/FLASH write them; keys without underscores, closer to BIDS labels). Register once per
    BEM method folder: `freesurfer.register(dm, root_key="watershed", root_path=...)`.
  - New `fascicle.extensions.mindmap` (default prefix `mm`), starting with mindmap-bem's
    surfaces: `mm-iskull`, `mm-oskull`, `mm-skin`.
  - Read as `(coords, faces)`; saving accepts `(coords, faces, volume_info)` and keeps it.
- [x] **Manuscript:** replace the local `BEM_DATATYPES` in `processing/common.py` with
  `freesurfer.register(...)` per MNE BEM method and `mindmap.register(...)`, and drop the
  `label("sub-01")` workarounds. Its `head` (`sub-XX-head.fif`) and `t1` keys aren't in
  fascicle. Done in the manuscript with 0.3.0: `watershed-head` and `fs-done`
  (`scripts/recon-all.done`) stay local `add_datatype` keys.
- [ ] **MNE `trans` and BEM head keys in the MNE extension.** The manuscript's coregistration
  writes `sub-XX/sub-XX-trans.fif` (in an MNE subjects dir) with a local `add_datatype` key
  (`coreg-trans`, read/written with `mne.read_trans` / `mne.write_trans(..., overwrite=True)`),
  and reads `bem/sub-XX-head-dense.fif` / `bem/sub-XX-head.fif` (`mne.read_bem_surfaces`) with
  local keys too (`coreg-head`, `watershed-head`). Suggested: `trans` (read/write) and `head`,
  `head-dense` (read-only) in `fascicle.extensions.mne`, on a subjects-dir root.

## Notes

- Version used by the manuscript: 0.3.0 from PyPI (pinned, `[mne]`), no longer an editable
  install of this checkout.
