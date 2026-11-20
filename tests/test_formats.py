"""Tests for fascicle.formats: registration and the default readers/writers."""

import numpy as np
import nibabel as nib
import pandas as pd
import pytest
from nibabel.streamlines import Tractogram

from fascicle import formats


class _FakeDM:
    """Minimal stand-in for a DatasetManager, just recording add_extension calls."""

    def __init__(self):
        self.ext_readers = {}
        self.ext_writers = {}

    def add_extension(self, ext, reader=None, writer=None):
        self.ext_readers[ext] = reader
        self.ext_writers[ext] = writer


# --- register ------------------------------------------------------------


def test_register_default_wires_all_extensions():
    dm = _FakeDM()

    formats.register(dm)

    assert set(dm.ext_readers) == set(dm.ext_writers) == set(formats.DEFAULT_EXTENSIONS)
    assert set(formats.DEFAULT_EXTENSIONS) == {
        ".nii",
        ".nii.gz",
        ".mgh",
        ".mgz",
        ".gii",
        ".trk",
        ".tck",
        ".json",
        ".bval",
        ".bvec",
        ".tsv",
        ".tsv.gz",
    }


def test_register_subset_only_wires_given_extensions():
    dm = _FakeDM()

    formats.register(dm, extensions=(".nii", ".mgz"))

    assert set(dm.ext_readers) == {".nii", ".mgz"}


def test_register_unknown_extension_raises():
    dm = _FakeDM()

    with pytest.raises(ValueError):
        formats.register(dm, extensions=(".bogus",))


def _wired(ext):
    dm = _FakeDM()
    formats.register(dm, extensions=(ext,))
    return dm.ext_readers[ext], dm.ext_writers[ext]


# --- images (.nii/.nii.gz/.mgh/.mgz/.gii) ----------------------------------


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz", ".mgh", ".mgz"])
def test_image_write_then_read_round_trip(tmp_path, suffix):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image_class = nib.MGHImage if suffix in (".mgh", ".mgz") else nib.Nifti1Image
    path = tmp_path / f"sub-01_T1w{suffix}"
    reader, writer = _wired(suffix)

    writer(image_class(data, np.eye(4)), path)
    loaded = reader(path)

    np.testing.assert_array_equal(loaded.get_fdata(), data)


def test_gifti_write_then_read_round_trip(tmp_path):
    darray = nib.gifti.GiftiDataArray(np.arange(12, dtype=np.float32))
    path = tmp_path / "sub-01_desc-foo.gii"
    reader, writer = _wired(".gii")

    writer(nib.gifti.GiftiImage(darrays=[darray]), path)
    loaded = reader(path)

    np.testing.assert_array_equal(loaded.darrays[0].data, darray.data)


# --- streamlines (.trk/.tck) -----------------------------------------------


@pytest.mark.parametrize("suffix", [".trk", ".tck"])
def test_streamlines_write_then_read_round_trip(tmp_path, suffix):
    streamlines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float),
        np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
    ]
    path = tmp_path / f"sub-01_desc-tracks{suffix}"
    reader, writer = _wired(suffix)

    writer(Tractogram(streamlines, affine_to_rasmm=np.eye(4)), path)
    loaded = reader(path)

    assert len(loaded.streamlines) == 2
    np.testing.assert_array_equal(loaded.streamlines[0], streamlines[0])


# --- JSON sidecars -----------------------------------------------------------


def test_json_write_then_read_round_trip(tmp_path):
    sidecar = {"RepetitionTime": 2.0, "TaskName": "rest", "SliceTiming": [0.0, 0.5, 1.0]}
    path = tmp_path / "sub-01_task-rest_bold.json"
    reader, writer = _wired(".json")

    writer(sidecar, path)

    assert reader(path) == sidecar
    assert path.read_text().endswith("}\n")


# --- diffusion gradient tables (.bval/.bvec) ---------------------------------


def test_bval_write_then_read_round_trip(tmp_path):
    bvals = np.array([0, 1000, 1000, 2000])
    path = tmp_path / "sub-01_dwi.bval"
    reader, writer = _wired(".bval")

    writer(bvals, path)

    assert path.read_text() == "0 1000 1000 2000\n"  # one space-separated row
    np.testing.assert_array_equal(reader(path), bvals)


def test_bvec_write_then_read_round_trip(tmp_path):
    bvecs = np.array([[0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 0.70710678]])
    path = tmp_path / "sub-01_dwi.bvec"
    reader, writer = _wired(".bvec")

    writer(bvecs, path)

    assert len(path.read_text().splitlines()) == 3
    np.testing.assert_array_almost_equal(reader(path), bvecs)


# --- tables (.tsv/.tsv.gz) ---------------------------------------------------


def test_tsv_write_then_read_round_trip_with_bids_na(tmp_path):
    events = pd.DataFrame(
        {"onset": [0.0, 1.5], "duration": [1.0, np.nan], "trial_type": ["go", "stop"]}
    )
    path = tmp_path / "sub-01_task-rest_events.tsv"
    reader, writer = _wired(".tsv")

    writer(events, path)

    assert path.read_text().splitlines() == [
        "onset\tduration\ttrial_type",
        "0.0\t1.0\tgo",
        "1.5\tn/a\tstop",
    ]
    pd.testing.assert_frame_equal(reader(path), events)


def test_tsv_reader_keeps_na_like_strings_other_than_bids_na(tmp_path):
    """Only BIDS's "n/a" is missing: a participant group called "NA" stays a string."""
    path = tmp_path / "participants.tsv"
    path.write_text("participant_id\tgroup\nsub-01\tNA\nsub-02\tn/a\n")
    reader, _ = _wired(".tsv")

    table = reader(path)

    assert table["group"].iloc[0] == "NA"
    assert pd.isna(table["group"].iloc[1])


def test_tsv_gz_is_compressed_and_round_trips(tmp_path):
    physio = pd.DataFrame({"cardiac": [0.1, 0.2, 0.3]})
    path = tmp_path / "sub-01_task-rest_physio.tsv.gz"
    reader, writer = _wired(".tsv.gz")

    writer(physio, path)

    assert path.read_bytes()[:2] == b"\x1f\x8b"  # gzip magic number
    pd.testing.assert_frame_equal(reader(path), physio)
