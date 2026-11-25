"""MNE-Python extension: raw MEG/EEG and source-analysis derivatives.

Raw recordings (unprefixed keys, in the raw BIDS dataset, read-only):

    meg, eeg                 `mne.io.Raw` (not preloaded), BIDS sidecars applied
    meg-info                 `mne.Info` only (sensors, digitisation, `dev_head_t`:
                             all a forward model needs), sidecars applied
    meg-file, eeg-file,      the same, exactly as stored in the file
    meg-info-file

These are found through the BIDS index (`DatasetManager.add_raw`), so any
format `mne.io.read_raw` reads works whatever its extension, and `run=1`
matches `run-01`:

    dm.load("01", "meg-info", session="meg", task="facerecognition", run=1)

"Sidecars applied" means the fields BIDS defines the same way in every
dataset, from the sidecars found by BIDS inheritance (`find_sidecar`):

- `channels.tsv` `type` -> channel types (through `BIDS_TO_MNE`), e.g. a
  channel the file calls EEG but the dataset calls HEOG becomes `eog`;
- `channels.tsv` `status` "bad" -> `info["bads"]`;
- `<meg|eeg>.json` `PowerLineFrequency` -> `info["line_freq"]`.

What changed is reported in `info["temp"]["bids_sidecars"]` (sidecar files
used, type changes, bads, line frequency, and channels skipped because the
file doesn't have them or their type is unknown), and logged at INFO on the
"fascicle" logger when something changed. Events, electrode positions and
coordinate systems depend on the dataset and are not applied.

Every sidecar also has its own key, taking the recording's entities:

    meg-channels, eeg-channels       channels.tsv     DataFrame
    meg-json, eeg-json               meg/eeg.json     dict
    meg-events, eeg-events           events.tsv       DataFrame
    meg-coordsystem, eeg-coordsystem coordsystem.json dict
    eeg-electrodes                   electrodes.tsv   DataFrame
    meg-headshape                    headshape.pos    DataFrame (label, x, y, z; units as stored)

Pass `raw=False` to `register` to skip all raw keys.

Derivatives (prefixed keys, in the extension's own folder): `src`, `fwd`,
`trans`, `ave`, `i-ave`, `v-ave`, `cov` and `stc` (keys `mne-src`, `mne-cov`, ... by
default) live under one derivatives root, named with
`fascicle.derivative_namer`, e.g.

    <root>/sub-01/cov/sub-01_task-rest_cov.fif
    <root>/sub-01/ses-meg/trans/sub-01_ses-meg_trans.fif             (trans)
    <root>/sub-01/ave/sub-01_task-rest_ave.fif                       (ave)
    <root>/sub-01/ave/sub-01_task-rest_signal-intracranial_ave.fif   (i-ave)
    <root>/sub-01/ave/sub-01_task-rest_signal-virtual_ave.fif        (v-ave)
    <root>/sub-01/stc/sub-01_task-rest_stc        (stem: -lh.stc/-rh.stc)

`i-ave` is the evoked response measured at intracranial contacts; `v-ave`
is the "virtual" one, estimated by projecting a source reconstruction onto
those contact locations. Both keep MNE's `_ave.fif` naming, told apart from
the scalp `ave` by a custom `signal` entity fixed per key (you never pass it).

Extra inputs are BIDS entities: `session`, `task`, `run`, `desc` (`src`/`fwd`/
`trans` take only `session`/`desc`). Unknown entities raise instead of being dropped.

Paths are built without looking at the disk, so the folder doesn't need to
exist yet (writing into a fresh derivatives folder works). For pybids
queries on top, also call `dm.add_derivatives(root_key, root_path)` once
the folder has a `dataset_description.json`.

The coregistration's head -> MRI transform is a derivative too: `trans`
(`mne-trans`), read/written as an `mne.transforms.Transform`. MNE takes it
from any path. The scalp surfaces MNE writes into a FreeSurfer subject's
`bem/` folder (`sub-XX-head.fif`, `sub-XX-head-dense.fif`) stay there, where
MNE looks for them: they are FreeSurfer extension keys (`fs-head`,
`fs-head-dense`).

`mne` is an optional dependency (`pip install "mindmap-fascicle[mne]"`): it's only
imported when a file is actually read or written.
"""

import logging
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from ..datatype import Datatype, RawDatatype
from ..extension import register_extension, register_raw
from ..formats import _read_json, _read_tsv
from ..naming import derivative_namer
from ..utils import find_sidecar

logger = logging.getLogger("fascicle")

_FILE_ENTITIES = ("session", "task", "run", "desc")
_ANAT_ENTITIES = ("session", "desc")


def _mne():
    try:
        import mne
    except ImportError as err:  # pragma: no cover - exercised only without mne
        raise ImportError(
            "Reading/writing MNE derivatives needs mne: pip install 'mindmap-fascicle[mne]'"
        ) from err
    return mne


# Recording files only: companions (.eeg/.vmrk, .fdt, .json) mustn't match.
# CTF .ds is a folder. BTi/4D (no extension) isn't read by mne.io.read_raw.
MEG_EXTENSIONS = (".fif", ".ds", ".con", ".sqd", ".bin")
EEG_EXTENSIONS = (".vhdr", ".set", ".edf", ".bdf")


def _read_raw(path):
    return _mne().io.read_raw(path, preload=False, verbose="warning")


def _read_info(path):
    """Read only the measurement info: directly for FIF, else from the raw header."""
    mne = _mne()
    if Path(path).suffix == ".fif":
        return mne.io.read_info(path, verbose="warning")
    return mne.io.read_raw(path, preload=False, verbose="warning").info


def _read_pos(path) -> pd.DataFrame:
    """Read a `headshape.pos` file: one `label x y z` row per point, units as stored.

    Lines that aren't `label x y z` (e.g. a leading point count) are skipped.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                rows.append((parts[0], *map(float, parts[1:])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["label", "x", "y", "z"])


# BIDS channel types (channels.tsv `type`, BIDS-MEG/EEG/iEEG appendix) -> MNE.
# `MEGGRAD` isn't in the current spec but older datasets (ds000117) use it.
BIDS_TO_MNE = {
    "MEGMAG": "mag",
    "MEGGRAD": "grad",
    "MEGGRADAXIAL": "grad",
    "MEGGRADPLANAR": "grad",
    "MEGREFMAG": "ref_meg",
    "MEGREFGRADAXIAL": "ref_meg",
    "MEGREFGRADPLANAR": "ref_meg",
    "MEGOTHER": "misc",
    "EEG": "eeg",
    "ECOG": "ecog",
    "SEEG": "seeg",
    "DBS": "dbs",
    "EOG": "eog",
    "HEOG": "eog",
    "VEOG": "eog",
    "ECG": "ecg",
    "EMG": "emg",
    "TRIG": "stim",
    "RESP": "resp",
    "GSR": "gsr",
    "TEMP": "temperature",
    "EYEGAZE": "eyegaze",
    "PUPIL": "pupil",
    "MISC": "misc",
    "AUDIO": "misc",
    "PD": "misc",
    "SYSCLOCK": "misc",
    "ADC": "misc",
    "DAC": "misc",
    "OTHER": "misc",
}


def _optional_sidecar(path, suffix, extension):
    try:
        return find_sidecar(path, suffix, extension)
    except FileNotFoundError:
        return None


def _apply_sidecars(data, path, suffix):
    """Apply the BIDS sidecar fields that mean the same in every dataset to `data`.

    `data` is an `mne.io.Raw` or `mne.Info`, read from the recording `path`.
    From `channels.tsv`: `type` sets channel types, `status` "bad" adds to
    bads. From `<suffix>.json`: `PowerLineFrequency` sets `line_freq`.
    Channels the file doesn't have and unknown types are skipped, not raised.

    The report goes to `info["temp"]["bids_sidecars"]` (not saved with the
    file); a summary is logged at INFO on the "fascicle" logger when
    something changed.
    """
    info = data if isinstance(data, _mne().Info) else data.info
    channels_file = _optional_sidecar(path, "channels", ".tsv")
    json_file = _optional_sidecar(path, suffix, ".json")
    report = {
        "files": {"channels": channels_file, "json": json_file},
        "types": {},
        "bads": [],
        "line_freq": None,
        "skipped": {},
    }

    if channels_file is not None:
        table = _read_tsv(channels_file)
        current = dict(zip(info["ch_names"], info.get_channel_types()))
        changes = {}
        for _, row in table.iterrows():
            name = row["name"]
            if name not in current:
                report["skipped"][name] = "not in recording"
                continue
            bids_type = row.get("type")
            # A missing column, BIDS "n/a" (read as NaN) or an empty cell.
            if not isinstance(bids_type, str) or not bids_type.strip():
                report["skipped"][name] = "no type in channels.tsv"
            elif (wanted := BIDS_TO_MNE.get(bids_type.strip().upper())) is None:
                report["skipped"][name] = f"unknown BIDS type {bids_type!r}"
            elif wanted != current[name]:
                changes[name] = wanted
            if str(row.get("status", "")).strip().lower() == "bad" and name not in info["bads"]:
                report["bads"].append(name)

        for name, wanted in changes.items():
            try:
                data.set_channel_types({name: wanted}, verbose="error")
            except Exception as err:  # MNE refuses some conversions
                report["skipped"][name] = f"type {current[name]} -> {wanted} refused: {err}"
            else:
                report["types"][name] = (current[name], wanted)
        if report["bads"]:
            info["bads"] = list(info["bads"]) + report["bads"]

    if json_file is not None:
        line_freq = _read_json(json_file).get("PowerLineFrequency")
        is_number = isinstance(line_freq, (int, float)) and not isinstance(line_freq, bool)
        if is_number and line_freq != info["line_freq"]:
            report["line_freq"] = (info["line_freq"], line_freq)
            info["line_freq"] = line_freq

    temp = info.get("temp")
    temp = temp if isinstance(temp, dict) else {}
    temp["bids_sidecars"] = report
    info["temp"] = temp

    if report["types"] or report["bads"] or report["line_freq"]:
        logger.info(
            "%s: applied BIDS sidecars (%d type changes, %d bads, line_freq %s)",
            Path(path).name, len(report["types"]), len(report["bads"]),
            "set" if report["line_freq"] else "unchanged",
        )
    return data


def _read_raw_bids(path, suffix):
    """`_read_raw`, then `_apply_sidecars`."""
    return _apply_sidecars(_read_raw(path), path, suffix)


def _read_info_bids(path, suffix):
    """`_read_info`, then `_apply_sidecars`."""
    return _apply_sidecars(_read_info(path), path, suffix)


def _read_trans(path):
    return _mne().read_trans(path, verbose="warning")


def _write_trans(trans, path):
    _mne().write_trans(path, trans, overwrite=True, verbose="warning")


def _read_bem_surfaces(path):
    """Read a BEM surface file as MNE's list of surface dicts.

    Used by the FreeSurfer extension's `head`/`head-dense` keys.
    """
    return _mne().read_bem_surfaces(path, verbose="warning")


def _read_src(path):
    return _mne().read_source_spaces(path, verbose="warning")


def _write_src(src, path):
    _mne().write_source_spaces(path, src, overwrite=True, verbose="warning")


def _read_fwd(path):
    return _mne().read_forward_solution(path, verbose="warning")


def _write_fwd(fwd, path):
    _mne().write_forward_solution(path, fwd, overwrite=True, verbose="warning")


def _read_evoked(path):
    return _mne().read_evokeds(path, verbose="warning")[0]


def _write_evoked(evoked, path):
    _mne().write_evokeds(path, evoked, overwrite=True, verbose="warning")


def _read_cov(path):
    cov = _mne().read_cov(path, verbose="warning")
    cov["data"] = cov["data"].astype("float")
    return cov


def _write_cov(cov, path):
    cov.save(path, overwrite=True, verbose="warning")


def _read_stc(stem):
    return _mne().read_source_estimate(stem)


def _write_stc(stc, stem):
    stc.save(stem, overwrite=True, verbose="warning")


def _stc_exists(stem: Path) -> bool:
    """A surface stc stem stands for a -lh.stc/-rh.stc pair (or one -stc.h5 / -vl.stc)."""
    pair = all(Path(f"{stem}-{hemi}.stc").exists() for hemi in ("lh", "rh"))
    return pair or any(Path(f"{stem}{ext}").exists() for ext in ("-stc.h5", "-vl.stc"))


def _dt(suffix, reader, writer, entities=_FILE_ENTITIES, extensions=(".fif",), exists=None,
        signal=None):
    fixed = {"signal": signal} if signal else None
    extra = ("signal",) if signal else ()
    namer = derivative_namer(suffix, suffix, extensions, entities, extra, fixed)
    return Datatype(namer, reader, writer, entities, exists)


DATATYPES: dict[str, Datatype] = {
    "src": _dt("src", _read_src, _write_src, _ANAT_ENTITIES),
    "fwd": _dt("fwd", _read_fwd, _write_fwd, _ANAT_ENTITIES),
    "trans": _dt("trans", _read_trans, _write_trans, _ANAT_ENTITIES),
    "ave": _dt("ave", _read_evoked, _write_evoked),
    "i-ave": _dt("ave", _read_evoked, _write_evoked, signal="intracranial"),
    "v-ave": _dt("ave", _read_evoked, _write_evoked, signal="virtual"),
    "cov": _dt("cov", _read_cov, _write_cov),
    "stc": _dt("stc", _read_stc, _write_stc, extensions=None, exists=_stc_exists),
}


def _sidecars(kind: str, extra: dict[str, tuple]) -> dict[str, RawDatatype]:
    """Sidecar keys `<kind>-<name>` of the `<kind>-file` recording."""
    common = {
        "channels": ("channels", None, (".tsv",)),
        "json": (kind, None, (".json",)),
        "events": ("events", None, (".tsv",)),
        "coordsystem": ("coordsystem", None, (".json",)),
        **extra,
    }
    return {
        f"{kind}-{name}": RawDatatype(suffix, reader, extensions, sidecar_of=f"{kind}-file")
        for name, (suffix, reader, extensions) in common.items()
    }


RAW: dict[str, RawDatatype] = {
    "meg-file": RawDatatype("meg", _read_raw, MEG_EXTENSIONS),
    "meg": RawDatatype("meg", partial(_read_raw_bids, suffix="meg"), MEG_EXTENSIONS),
    "meg-info-file": RawDatatype("meg", _read_info, MEG_EXTENSIONS),
    "meg-info": RawDatatype("meg", partial(_read_info_bids, suffix="meg"), MEG_EXTENSIONS),
    "eeg-file": RawDatatype("eeg", _read_raw, EEG_EXTENSIONS),
    "eeg": RawDatatype("eeg", partial(_read_raw_bids, suffix="eeg"), EEG_EXTENSIONS),
    **_sidecars("meg", {"headshape": ("headshape", _read_pos, (".pos",))}),
    **_sidecars("eeg", {"electrodes": ("electrodes", None, (".tsv",))}),
}


def register(
    dm,
    root_key: str = "mne",
    root_path: str | Path | None = None,
    namespace: str | None = None,
    readers: dict[str, Any] | None = None,
    writers: dict[str, Any] | None = None,
    raw: bool = True,
) -> None:
    """Register raw MEG/EEG and MNE source-analysis datatypes on `dm`.

    Parameters
    ----------
    dm: fascicle.DatasetManager
    root_key: str
        Root key for the MNE derivatives folder.
    root_path: str | Path, optional
        Path to add as `root_key` (needn't exist yet). If omitted,
        `root_key` must already exist in `dm.roots`.
    namespace: str, optional
        Key prefix; defaults to `root_key`.
    readers, writers: dict, optional
        Bare derivative name -> replacement function. To swap a raw
        reader, use `dm.set_reader("meg", ...)` after registering.
    raw: bool
        Also register the raw `meg`, `eeg` and `meg-info` keys.

    Raises
    ------
    KeyError
        If `root_path` is omitted and `root_key` isn't already registered.
    ValueError
        If a raw key is already registered with something else.
    """
    if raw:
        register_raw(dm, RAW)
    register_extension(
        dm,
        DATATYPES,
        root_key=root_key,
        root_path=root_path,
        namespace=namespace,
        readers=readers,
        writers=writers,
    )

