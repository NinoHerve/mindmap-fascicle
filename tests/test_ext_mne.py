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
