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
- [ ] **The `run` entity isn't zero-padded to match the dataset.** Fixed for keys registered
  with `add_raw` (`meg`, `eeg`, `meg-info`): the index stores runs as integers. Still open for
  `bids_namer` keys such as `T1w`.
  - `get_path(..., run=1)` builds `..._run-1_meg.fif`, but ds000117 names it `run-01`.
    The path is wrong, and the error only says "No file for 'meg'".
  - `run="01"` works, so it's a usability issue, not wrong data: BIDS allows both forms,
    and pybids `build_path` uses the value as given.
  - Suggested: when the built path doesn't exist, look up the run in the layout, which
    stores runs as integers, and use the indexed file. Otherwise mention the padding in
    the error message.

## Notes

- Version tested: 0.1.0 (`pyproject.toml`). The manuscript keeps an editable install of this
  checkout until it pins a release.
