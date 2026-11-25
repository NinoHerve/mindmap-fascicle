"""Tests for fascicle.extensions.mne.

Path naming is tested without mne; read/write round-trips need it.
"""

import numpy as np
import pytest

from fascicle import DatasetManager
from fascicle.extensions import mne as mne_ext


# --- path naming (no mne needed) -------------------------------------------


@pytest.mark.parametrize(
    "suffix, entities, expected",
    [
        ("cov", {}, "sub-01/cov/sub-01_cov.fif"),
        ("cov", {"task": "rest"}, "sub-01/cov/sub-01_task-rest_cov.fif"),
        ("ave", {"session": "a", "run": 2}, "sub-01/ses-a/ave/sub-01_ses-a_run-2_ave.fif"),
        ("i-ave", {"desc": "x"}, "sub-01/ave/sub-01_signal-intracranial_desc-x_ave.fif"),
        ("v-ave", {"task": "rest"}, "sub-01/ave/sub-01_task-rest_signal-virtual_ave.fif"),
        ("src", {"desc": "oct6"}, "sub-01/src/sub-01_desc-oct6_src.fif"),
        ("fwd", {}, "sub-01/fwd/sub-01_fwd.fif"),
        ("stc", {"task": "rest"}, "sub-01/stc/sub-01_task-rest_stc"),
    ],
)
def test_paths(tmp_path, suffix, entities, expected):
    """Same paths as the earlier hand-written pybids pattern file, except
    i-ave/v-ave, which now keep MNE's `_ave.fif` naming plus a signal entity."""
    namer = mne_ext.DATATYPES[suffix].namer

    assert namer(tmp_path, "sub-01", **entities) == tmp_path / expected


def test_rejects_entity_the_suffix_does_not_take(tmp_path):
    with pytest.raises(ValueError, match="task"):
        mne_ext.DATATYPES["fwd"].namer(tmp_path, "01", task="rest")


def test_rejects_extension_other_than_fif(tmp_path):
    with pytest.raises(ValueError, match=".h5"):
        mne_ext.DATATYPES["cov"].namer(tmp_path, "01", extension=".h5")


@pytest.mark.parametrize("key", ["i-ave", "v-ave"])
def test_signal_is_fixed_not_passed(tmp_path, key):
    with pytest.raises(ValueError, match="signal"):
        mne_ext.DATATYPES[key].namer(tmp_path, "01", signal="virtual")


def test_plain_ave_does_not_take_signal(tmp_path):
    with pytest.raises(ValueError, match="signal"):
        mne_ext.DATATYPES["ave"].namer(tmp_path, "01", signal="intracranial")


def test_inputs_match_accepted_entities():
    assert mne_ext.DATATYPES["src"].inputs == ("session", "desc")
    assert mne_ext.DATATYPES["cov"].inputs == ("session", "task", "run", "desc")


@pytest.fixture
def dm(bids_root, tmp_path):
    manager = DatasetManager(bids_root)
    mne_ext.register(manager, root_path=tmp_path / "derivatives" / "mne")
    return manager


def test_register_keys(dm):
    expected = {f"mne-{name}" for name in ("src", "fwd", "ave", "i-ave", "v-ave", "cov", "stc")}

    assert expected <= set(dm.registry)
    assert {"meg", "eeg", "meg-info"} <= set(dm.registry)


def test_raw_keys_are_on_raw_root_and_read_only(dm, bids_root):
    for key in ("meg", "eeg", "meg-info"):
        info = dm.describe(key)
        assert info["root_key"] == "raw"
        assert info["root"] == bids_root
        assert info["writer"] == "READ_ONLY"


def test_register_raw_false_skips_raw_keys(bids_root, tmp_path):
    manager = DatasetManager(bids_root)
    mne_ext.register(manager, root_path=tmp_path / "mne", raw=False)

    assert not {"meg", "eeg", "meg-info"} & set(manager.registry)


def test_register_twice_under_two_roots_keeps_one_set_of_raw_keys(dm, tmp_path):
    mne_ext.register(dm, root_key="mne2", root_path=tmp_path / "mne2")

    assert "mne2-cov" in dm.registry
    assert "mne2-meg" not in dm.registry


def test_eeg_finds_header_not_companions(raw_root, tmp_path):
    manager = DatasetManager(raw_root)
    mne_ext.register(manager, root_path=tmp_path / "mne")

    assert manager.get_path("01", "eeg", task="x").name == "sub-01_task-x_eeg.vhdr"


def test_meg_finds_ctf_folder(raw_root, tmp_path):
    manager = DatasetManager(raw_root)
    mne_ext.register(manager, root_path=tmp_path / "mne")

    assert manager.get_path("01", "meg", session="a", task="y").suffix == ".ds"


def test_register_does_not_require_existing_folder(dm, tmp_path):
    assert not (tmp_path / "derivatives").exists()
    assert dm.get_path("01", "mne-cov").name == "sub-01_cov.fif"


# --- round-trips (need mne) --------------------------------------------------


@pytest.fixture
def mne():
    return pytest.importorskip("mne")


def _info(mne):
    return mne.create_info(["a", "b", "c"], sfreq=100.0, ch_types="eeg")


def test_cov_round_trip(dm, mne):
    data = np.diag([1.0, 2.0, 3.0])
    cov = mne.Covariance(data, ["a", "b", "c"], bads=[], projs=[], nfree=10)

    path = dm.save(cov, "01", "mne-cov", task="rest")
    loaded = dm.load("01", "mne-cov", task="rest")

    assert path.name == "sub-01_task-rest_cov.fif"
    assert loaded["data"].dtype == float
    np.testing.assert_array_almost_equal(loaded.data, data)
    assert loaded.ch_names == ["a", "b", "c"]


@pytest.mark.parametrize("key", ["mne-ave", "mne-i-ave", "mne-v-ave"])
def test_evoked_round_trip(dm, mne, key):
    data = np.arange(15, dtype=float).reshape(3, 5) * 1e-6
    evoked = mne.EvokedArray(data, _info(mne), tmin=0.0)

    dm.save(evoked, "01", key, session="a")
    loaded = dm.load("01", key, session="a")

    np.testing.assert_array_almost_equal(loaded.data, data)


def test_stc_round_trip_and_stem_exists_check(dm, mne):
    vertices = [np.array([0, 1]), np.array([0])]
    data = np.arange(6, dtype=float).reshape(3, 2)
    stc = mne.SourceEstimate(data, vertices, tmin=0.0, tstep=0.01)

    stem = dm.save(stc, "01", "mne-stc")
    loaded = dm.load("01", "mne-stc")

    assert not stem.exists()  # only the -lh.stc/-rh.stc pair is on disk
    assert stem.with_name(stem.name + "-lh.stc").exists()
    np.testing.assert_array_almost_equal(loaded.data, data)
    with pytest.raises(FileExistsError):
        dm.save(stc, "01", "mne-stc")


def test_load_missing_stc_raises_file_not_found(dm, mne):
    with pytest.raises(FileNotFoundError):
        dm.load("01", "mne-stc", task="none")


@pytest.fixture
def meg_dm(bids_root, tmp_path, mne):
    """A manager over `bids_root` plus one real FIF MEG run, `run-01`."""
    info = mne.create_info(["MEG 0111", "MEG 0112", "EEG 001"], 100.0, ["mag", "grad", "eeg"])
    data = np.arange(30, dtype=float).reshape(3, 10) * 1e-12
    meg = bids_root / "sub-01" / "ses-a" / "meg"
    meg.mkdir(parents=True)
    raw = mne.io.RawArray(data, info, verbose="warning")
    raw.save(meg / "sub-01_ses-a_task-x_run-01_meg.fif", verbose="warning")

    manager = DatasetManager(bids_root)
    mne_ext.register(manager, root_path=tmp_path / "derivatives" / "mne")
    return manager, data


def test_meg_load_is_lazy_raw(meg_dm):
    dm, data = meg_dm

    raw = dm.load("01", "meg", session="a", task="x", run=1)

    assert raw.preload is False
    np.testing.assert_array_almost_equal(raw.get_data(), data)


def test_meg_info_reads_only_info(meg_dm, mne):
    dm, _ = meg_dm

    info = dm.load("01", "meg-info", session="a", task="x", run=1)

    assert isinstance(info, mne.Info)
    assert info["ch_names"] == ["MEG 0111", "MEG 0112", "EEG 001"]


def test_meg_info_falls_back_to_raw_header_for_other_formats(monkeypatch, mne, tmp_path):
    class FakeRaw:
        info = "header info"

    calls = []

    def fake_read_raw(path, preload, verbose):
        calls.append(preload)
        return FakeRaw()

    monkeypatch.setattr(mne.io, "read_raw", fake_read_raw)

    assert mne_ext._read_info(tmp_path / "sub-01_task-x_meg.ds") == "header info"
    assert calls == [False]


def test_meg_save_refused(meg_dm):
    dm, _ = meg_dm

    with pytest.raises(TypeError, match="read-only"):
        dm.save(object(), "01", "meg", session="a", task="x", run=1, overwrite=True)


# --- BIDS sidecars ------------------------------------------------------------


def _write_meg(bids_root, mne, line_freq=None):
    """One FIF run with EEG061 mislabelled as EEG, as in ds000117."""
    info = mne.create_info(["MEG0111", "EEG001", "EEG061"], 100.0, ["mag", "eeg", "eeg"])
    if line_freq is not None:
        info["line_freq"] = line_freq
    meg = bids_root / "sub-01" / "ses-a" / "meg"
    meg.mkdir(parents=True)
    raw = mne.io.RawArray(np.zeros((3, 10)), info, verbose="warning")
    raw.save(meg / "sub-01_ses-a_task-x_run-01_meg.fif", verbose="warning")
    return meg


@pytest.fixture
def sidecar_dm(bids_root, tmp_path, mne):
    """MEG run plus session-level channels.tsv/meg.json and run-level sidecars."""
    meg = _write_meg(bids_root, mne, line_freq=60.0)
    session = meg.parent
    (session / "sub-01_ses-a_task-x_channels.tsv").write_text(
        "name\ttype\tunits\tstatus\n"
        "MEG0111\tMEGMAG\tT\tgood\n"
        "EEG001\tEEG\tV\tbad\n"
        "EEG061\tHEOG\tV\tgood\n"
        "CHPI001\tMISC\tn/a\tgood\n"  # listed, but not in the file
    )
    (session / "sub-01_ses-a_task-x_meg.json").write_text('{"PowerLineFrequency": 50}')
    (meg / "sub-01_ses-a_coordsystem.json").write_text('{"MEGCoordinateUnits": "mm"}')
    (meg / "sub-01_ses-a_headshape.pos").write_text("2\nHP1 1.0 2.0 3.0\nEEG1 4 5 6\n")
    (meg / "sub-01_ses-a_task-x_run-01_events.tsv").write_text("onset\tduration\n0.5\t0\n")

    eeg = bids_root / "sub-01" / "eeg"
    eeg.mkdir()
    (eeg / "sub-01_task-x_eeg.vhdr").touch()
    (eeg / "sub-01_electrodes.tsv").write_text("name\tx\ty\tz\nFz\t0\t1\t2\n")

    manager = DatasetManager(bids_root)
    mne_ext.register(manager, root_path=tmp_path / "derivatives" / "mne")
    return manager


RUN = {"session": "a", "task": "x", "run": 1}


def test_all_raw_keys_registered(dm):
    expected = {
        "meg", "meg-file", "meg-info", "meg-info-file", "eeg", "eeg-file",
        "meg-channels", "meg-json", "meg-events", "meg-coordsystem", "meg-headshape",
        "eeg-channels", "eeg-json", "eeg-events", "eeg-coordsystem", "eeg-electrodes",
    }

    assert expected <= set(dm.registry)


def test_meg_applies_channel_types_bads_and_line_freq(sidecar_dm):
    raw = sidecar_dm.load("01", "meg", **RUN)

    assert raw.get_channel_types() == ["mag", "eeg", "eog"]
    assert raw.info["bads"] == ["EEG001"]
    assert raw.info["line_freq"] == 50


def test_meg_report_lists_changes_and_skips(sidecar_dm):
    report = sidecar_dm.load("01", "meg", **RUN).info["temp"]["bids_sidecars"]

    assert report["files"]["channels"].name == "sub-01_ses-a_task-x_channels.tsv"
    assert report["files"]["json"].name == "sub-01_ses-a_task-x_meg.json"
    assert report["types"] == {"EEG061": ("eeg", "eog")}
    assert report["bads"] == ["EEG001"]
    assert report["line_freq"] == (60.0, 50)
    assert report["skipped"] == {"CHPI001": "not in recording"}


def test_meg_logs_when_something_changed(sidecar_dm, caplog):
    with caplog.at_level("INFO", logger="fascicle"):
        sidecar_dm.load("01", "meg", **RUN)

    assert "1 type changes, 1 bads" in caplog.text


def test_meg_file_is_untouched(sidecar_dm):
    raw = sidecar_dm.load("01", "meg-file", **RUN)

    assert raw.get_channel_types() == ["mag", "eeg", "eeg"]
    assert raw.info["bads"] == []
    assert "bids_sidecars" not in (raw.info.get("temp") or {})


def test_meg_info_applies_sidecars_and_info_file_does_not(sidecar_dm, mne):
    info = sidecar_dm.load("01", "meg-info", **RUN)
    plain = sidecar_dm.load("01", "meg-info-file", **RUN)

    assert isinstance(info, mne.Info)
    assert info.get_channel_types() == ["mag", "eeg", "eog"]
    assert info["bads"] == ["EEG001"]
    assert plain.get_channel_types() == ["mag", "eeg", "eeg"]


def test_meg_without_sidecars_loads_silently(bids_root, tmp_path, mne, caplog):
    _write_meg(bids_root, mne)
    manager = DatasetManager(bids_root)
    mne_ext.register(manager, root_path=tmp_path / "mne")

    with caplog.at_level("INFO", logger="fascicle"):
        raw = manager.load("01", "meg", **RUN)

    report = raw.info["temp"]["bids_sidecars"]
    assert report["files"] == {"channels": None, "json": None}
    assert raw.get_channel_types() == ["mag", "eeg", "eeg"]
    assert caplog.text == ""


def test_unknown_bids_type_is_skipped(sidecar_dm, bids_root):
    channels = bids_root / "sub-01" / "ses-a" / "sub-01_ses-a_task-x_channels.tsv"
    channels.write_text("name\ttype\nEEG061\tNOTATYPE\n")

    raw = sidecar_dm.load("01", "meg", **RUN)

    assert raw.get_channel_types()[2] == "eeg"
    assert "NOTATYPE" in raw.info["temp"]["bids_sidecars"]["skipped"]["EEG061"]


def test_sidecar_keys_load(sidecar_dm):
    channels = sidecar_dm.load("01", "meg-channels", **RUN)
    assert list(channels["name"]) == ["MEG0111", "EEG001", "EEG061", "CHPI001"]
    assert sidecar_dm.load("01", "meg-json", **RUN) == {"PowerLineFrequency": 50}
    assert sidecar_dm.load("01", "meg-coordsystem", session="a") == {"MEGCoordinateUnits": "mm"}
    assert len(sidecar_dm.load("01", "meg-events", **RUN)) == 1
    assert list(sidecar_dm.load("01", "eeg-electrodes", task="x")["name"]) == ["Fz"]


def test_headshape_pos_skips_count_line(sidecar_dm):
    points = sidecar_dm.load("01", "meg-headshape", session="a")

    assert list(points.columns) == ["label", "x", "y", "z"]
    assert list(points["label"]) == ["HP1", "EEG1"]
    assert points.loc[1, ["x", "y", "z"]].tolist() == [4.0, 5.0, 6.0]


# --- _apply_sidecars on its own (EEG, edge cases) ------------------------------


def _eeg_case(tmp_path, channels, eeg_json=None, meg_json=None):
    """A fake EEG recording path with sidecars next to it (the file itself isn't read)."""
    root = tmp_path / "ds"
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (root / "dataset_description.json").write_text("{}")
    (eeg / "sub-01_task-x_channels.tsv").write_text("name\ttype\tstatus\n" + channels)
    for kind, content in (("eeg", eeg_json), ("meg", meg_json)):
        if content is not None:
            (eeg / f"sub-01_task-x_{kind}.json").write_text(content)
    return eeg / "sub-01_task-x_eeg.vhdr"


def _eeg_info(mne, **fields):
    info = mne.create_info(["Fz", "HEOG"], 100.0, "eeg")
    for key, value in fields.items():
        info[key] = value
    return info


def test_eeg_sidecars_use_eeg_json(tmp_path, mne):
    path = _eeg_case(
        tmp_path, "Fz\tEEG\tgood\nHEOG\teog\tgood\n",  # lower-case type
        eeg_json='{"PowerLineFrequency": 50}', meg_json='{"PowerLineFrequency": 60}',
    )

    info = mne_ext._apply_sidecars(_eeg_info(mne), path, "eeg")
    report = info["temp"]["bids_sidecars"]

    assert info.get_channel_types() == ["eeg", "eog"]
    assert info["line_freq"] == 50
    assert report["files"]["json"].name == "sub-01_task-x_eeg.json"


def test_na_channel_type_is_not_reported_as_unknown(tmp_path, mne):
    path = _eeg_case(tmp_path, "Fz\tn/a\tbad\nHEOG\tEOG\tgood\n")

    info = mne_ext._apply_sidecars(_eeg_info(mne), path, "eeg")
    report = info["temp"]["bids_sidecars"]

    assert report["skipped"] == {"Fz": "no type in channels.tsv"}
    assert info.get_channel_types() == ["eeg", "eog"]
    assert info["bads"] == ["Fz"]  # status still applied


def test_missing_type_column_is_not_reported_as_unknown(tmp_path, mne):
    path = _eeg_case(tmp_path, "")
    (path.parent / "sub-01_task-x_channels.tsv").write_text("name\tstatus\nFz\tgood\n")

    report = mne_ext._apply_sidecars(_eeg_info(mne), path, "eeg")["temp"]["bids_sidecars"]

    assert report["skipped"] == {"Fz": "no type in channels.tsv"}


def test_refused_type_change_is_recorded_and_others_still_apply(tmp_path, mne, monkeypatch):
    path = _eeg_case(tmp_path, "Fz\tECG\tgood\nHEOG\tEOG\tgood\n")
    original = mne.Info.set_channel_types

    def refuse_fz(self, mapping, *args, **kwargs):
        if "Fz" in mapping:
            raise ValueError("not allowed")
        return original(self, mapping, *args, **kwargs)

    monkeypatch.setattr(mne.Info, "set_channel_types", refuse_fz)

    info = mne_ext._apply_sidecars(_eeg_info(mne), path, "eeg")
    report = info["temp"]["bids_sidecars"]

    assert "refused: not allowed" in report["skipped"]["Fz"]
    assert report["types"] == {"HEOG": ("eeg", "eog")}
    assert info.get_channel_types() == ["eeg", "eog"]


def test_bad_already_in_file_is_not_added_twice(tmp_path, mne):
    path = _eeg_case(tmp_path, "Fz\tEEG\tbad\nHEOG\tEEG\tgood\n")

    info = mne_ext._apply_sidecars(_eeg_info(mne, bads=["Fz"]), path, "eeg")

    assert info["bads"] == ["Fz"]
    assert info["temp"]["bids_sidecars"]["bads"] == []


def test_existing_temp_entries_are_kept(tmp_path, mne):
    path = _eeg_case(tmp_path, "Fz\tEEG\tgood\n")

    info = mne_ext._apply_sidecars(_eeg_info(mne, temp={"mine": 1}), path, "eeg")

    assert info["temp"]["mine"] == 1
    assert "bids_sidecars" in info["temp"]


@pytest.mark.parametrize("value", ['"n/a"', "true"])
def test_non_numeric_line_freq_is_ignored(tmp_path, mne, value):
    path = _eeg_case(tmp_path, "Fz\tEEG\tgood\n", eeg_json=f'{{"PowerLineFrequency": {value}}}')

    info = mne_ext._apply_sidecars(_eeg_info(mne), path, "eeg")

    assert info["line_freq"] is None
    assert info["temp"]["bids_sidecars"]["line_freq"] is None


def test_matching_line_freq_is_not_a_change(tmp_path, mne, caplog):
    path = _eeg_case(tmp_path, "Fz\tEEG\tgood\n", eeg_json='{"PowerLineFrequency": 50}')

    with caplog.at_level("INFO", logger="fascicle"):
        info = mne_ext._apply_sidecars(_eeg_info(mne, line_freq=50.0), path, "eeg")

    assert info["temp"]["bids_sidecars"]["line_freq"] is None
    assert caplog.text == ""


def test_read_pos_skips_non_numeric_lines(tmp_path):
    path = tmp_path / "sub-01_headshape.pos"
    path.write_text("HP1 a b c\nHP2 1 2 3\nnot a point\n")

    points = mne_ext._read_pos(path)

    assert list(points["label"]) == ["HP2"]


# --- subjects-dir keys: trans, head, head-dense --------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("trans", "sub-01/sub-01-trans.fif"),
        ("head", "sub-01/bem/sub-01-head.fif"),
        ("head-dense", "sub-01/bem/sub-01-head-dense.fif"),
    ],
)
@pytest.mark.parametrize("subject", ["01", "sub-01"])
def test_subjects_dir_paths(tmp_path, name, expected, subject):
    namer = mne_ext.SUBJECTS_DIR_DATATYPES[name].namer

    assert namer(tmp_path, subject) == tmp_path / expected


@pytest.fixture
def subjects_dm(bids_root, tmp_path):
    """FreeSurfer keys and the MNE subjects-dir keys on one (not yet existing) root."""
    from fascicle.extensions import freesurfer

    manager = DatasetManager(bids_root)
    freesurfer.register(manager, root_path=tmp_path / "subjects")
    mne_ext.register_subjects_dir(manager)
    return manager


def test_subjects_dir_keys_share_the_freesurfer_root(subjects_dm, tmp_path):
    assert {"fs-trans", "fs-head", "fs-head-dense", "fs-t1", "fs-iskull"} <= set(subjects_dm.registry)
    for key in ("fs-trans", "fs-head", "fs-head-dense"):
        assert subjects_dm.describe(key)["root"] == tmp_path / "subjects"


def test_subjects_dir_paths_need_no_folder(subjects_dm, tmp_path):
    assert not (tmp_path / "subjects").exists()
    assert subjects_dm.get_path("sub-01", "fs-trans") == tmp_path / "subjects/sub-01/sub-01-trans.fif"


def test_subjects_dir_own_root(bids_root, tmp_path):
    manager = DatasetManager(bids_root)
    mne_ext.register_subjects_dir(manager, root_key="coreg", root_path=tmp_path / "coreg")

    assert {"coreg-trans", "coreg-head", "coreg-head-dense"} <= set(manager.registry)
    assert manager.describe("coreg-trans")["root"] == tmp_path / "coreg"


def test_subjects_dir_missing_root_raises(bids_root):
    with pytest.raises(KeyError, match="missing"):
        mne_ext.register_subjects_dir(DatasetManager(bids_root), root_key="missing")


def test_head_keys_are_read_only_and_trans_is_not(subjects_dm):
    assert subjects_dm.describe("fs-head")["writer"] == "READ_ONLY"
    assert subjects_dm.describe("fs-head-dense")["writer"] == "READ_ONLY"
    assert subjects_dm.describe("fs-trans")["writer"] != "READ_ONLY"


def test_trans_round_trip(subjects_dm, mne, tmp_path):
    matrix = np.eye(4)
    matrix[:3, 3] = [0.01, -0.02, 0.03]
    trans = mne.transforms.Transform("head", "mri", matrix)

    path = subjects_dm.save(trans, "01", "fs-trans")
    loaded = subjects_dm.load("sub-01", "fs-trans")

    assert path == tmp_path / "subjects/sub-01/sub-01-trans.fif"
    assert (loaded["from"], loaded["to"]) == (trans["from"], trans["to"])
    np.testing.assert_array_almost_equal(loaded["trans"], matrix)

    with pytest.raises(FileExistsError):
        subjects_dm.save(trans, "01", "fs-trans")
    matrix[:3, 3] = 0
    subjects_dm.save(mne.transforms.Transform("head", "mri", matrix), "01", "fs-trans", overwrite=True)
    np.testing.assert_array_almost_equal(subjects_dm.load("01", "fs-trans")["trans"], np.eye(4))


def test_head_dense_reads_bem_surfaces_and_refuses_save(subjects_dm, mne):
    from mne.io.constants import FIFF

    rr = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=float)
    tris = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    surf = dict(id=FIFF.FIFFV_BEM_SURF_ID_HEAD, sigma=0.33, np=len(rr), ntri=len(tris),
                coord_frame=FIFF.FIFFV_COORD_MRI, rr=rr, tris=tris)
    path = subjects_dm.get_path("01", "fs-head-dense")
    path.parent.mkdir(parents=True)
    mne.write_bem_surfaces(path, [surf], verbose="warning")

    loaded = subjects_dm.load("01", "fs-head-dense")

    assert isinstance(loaded, list) and len(loaded) == 1
    np.testing.assert_array_almost_equal(loaded[0]["rr"], rr)
    with pytest.raises(TypeError, match="read-only"):
        subjects_dm.save(loaded, "01", "fs-head-dense", overwrite=True)
