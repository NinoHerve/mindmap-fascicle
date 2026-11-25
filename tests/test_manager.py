"""Tests for fascicle.manager.DatasetManager."""

import json

import nibabel as nib
import numpy as np
import pytest

from fascicle import DatasetManager, Datatype


# --- construction: raw_root / config priority -----------------------------


def test_missing_raw_root_raises():
    with pytest.raises(KeyError):
        DatasetManager()


def test_raw_root_alone(bids_root):
    dm = DatasetManager(bids_root)

    assert dm.get_subjects() == ["01", "02"]


def test_config_dir_raw_alone(bids_root):
    dm = DatasetManager(config={"DIR_RAW": bids_root})

    assert dm.get_subjects() == ["01", "02"]


def test_raw_root_overrides_config_dir_raw(bids_root, tmp_path):
    other_root = tmp_path / "other_bids"
    other_root.mkdir()
    (other_root / "dataset_description.json").write_text(
        json.dumps({"Name": "Other dataset", "BIDSVersion": "1.8.0"})
    )
    anat = other_root / "sub-99" / "anat"
    anat.mkdir(parents=True)
    img = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
    nib.save(img, anat / "sub-99_T1w.nii.gz")

    dm = DatasetManager(raw_root=bids_root, config={"DIR_RAW": other_root})

    assert dm.get_subjects() == ["01", "02"]


# --- get_subjects / load / get_path ---------------------------------------


def test_get_subjects(bids_root):
    dm = DatasetManager(bids_root)

    assert dm.get_subjects() == dm.subjects


def test_get_path_and_load_t1w(bids_root):
    dm = DatasetManager(bids_root)

    path = dm.get_path("01", "T1w")
    assert str(path).endswith("sub-01_T1w.nii.gz")

    img = dm.load("01", "T1w")
    assert img.shape == (2, 3, 4)


def test_load_unknown_subject_raises_value_error(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(ValueError):
        dm.load("99", "T1w")


def test_load_unknown_suffix_raises_key_error(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError):
        dm.load("01", "bogus")


def test_get_path_unknown_suffix_raises_key_error(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError):
        dm.get_path("01", "bogus")


def test_load_by_extension_unknown_extension_raises_value_error(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    mystery_file = tmp_path / "data.mystery"
    mystery_file.write_text("data")

    with pytest.raises(ValueError):
        dm.load_by_extension(mystery_file)


# --- registry mutation: add_root / add_datatype / add_extension ----------


def _text_root(tmp_path, dm):
    custom_root = tmp_path / "custom"
    custom_root.mkdir()
    (custom_root / "sub-01_custom.txt").write_text("hello")
    dm.add_root("custom", custom_root)
    return custom_root


def _custom_namer(root, subject, **inputs):
    return root / f"sub-{subject}_custom.txt"


def _read_text(path):
    return path.read_text()


def _write_text(data, path):
    path.write_text(data)


def test_add_root_and_add_datatype_custom_key(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    _text_root(tmp_path, dm)

    dm.add_datatype("custom", "custom", _custom_namer, _read_text)

    assert dm.load("01", "custom") == "hello"


def test_add_datatype_returns_key_and_builds_datatype(bids_root):
    dm = DatasetManager(bids_root)

    key = dm.add_datatype("plain", None, _custom_namer, inputs=("hemi",))

    assert key == "plain"
    assert dm.registry["plain"] == Datatype(_custom_namer, inputs=("hemi",))


def test_add_datatype_namespace_prefixes_key(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    _text_root(tmp_path, dm)

    key = dm.add_datatype("custom", "custom", _custom_namer, _read_text, namespace="ns")

    assert key == "ns-custom"
    assert "ns-custom" in dm.registry
    assert "custom" not in dm.registry
    assert dm.load("01", "ns-custom") == "hello"


def test_add_registers_ready_made_datatype(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    _text_root(tmp_path, dm)

    dm.add("custom", Datatype(_custom_namer, _read_text, root_key="custom"), namespace="ns")

    assert dm.load("01", "ns-custom") == "hello"


def test_namer_receives_inputs(bids_root):
    dm = DatasetManager(bids_root)
    seen = {}

    def namer(root, subject, **inputs):
        seen.update(inputs, subject=subject)
        return "some/path"

    dm.add_datatype("thing", None, namer)
    dm.get_path("01", "thing", hemi="lh")

    assert seen == {"subject": "01", "hemi": "lh"}


def test_get_path_unknown_key_error_lists_registered_datatypes(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError, match=r"T1w"):
        dm.get_path("01", "bogus")


def test_load_unknown_key_error_lists_registered_datatypes(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError, match=r"T1w"):
        dm.load("01", "bogus")


def test_bids_namer_for_bold_threads_entities(bids_root):
    """A second BIDS suffix (bold) gets its namer from bids_namer, threading
    extra entities (task) through get_path/load and loading 4D data."""
    func = bids_root / "sub-01" / "func"
    func.mkdir(parents=True)
    bold_data = np.zeros((2, 3, 4, 5), dtype=np.float32)
    nib.save(nib.Nifti1Image(bold_data, np.eye(4)), func / "sub-01_task-rest_bold.nii.gz")

    dm = DatasetManager(bids_root)
    dm.add_datatype("bold", "raw", dm.bids_namer("bold"))

    path = dm.get_path("01", "bold", task="rest")
    assert str(path).endswith("sub-01_task-rest_bold.nii.gz")

    loaded = dm.load("01", "bold", task="rest")
    assert loaded.shape == (2, 3, 4, 5)


def test_t1w_is_a_datatype_dispatching_on_extension(bids_root):
    dm = DatasetManager(bids_root)

    entry = dm.registry["T1w"]

    assert isinstance(entry, Datatype)
    assert entry.root_key == "raw"
    assert entry.reader is None and entry.writer is None


def test_add_datatype_root_key_none(bids_root, tmp_path):
    dm = DatasetManager(bids_root)

    standalone_file = tmp_path / "standalone.txt"
    standalone_file.write_text("no shared root")

    def namer(root, subject, **inputs):
        assert root is None
        return standalone_file

    dm.add_datatype("standalone", None, namer, _read_text)

    assert dm.load("01", "standalone") == "no shared root"


def test_add_extension(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    custom_file = tmp_path / "data.custom"
    custom_file.write_text("42")

    dm.add_extension(".custom", lambda p: int(p.read_text()), lambda d, p: p.write_text(str(d)))

    assert dm.load_by_extension(custom_file) == 42
    dm.save_by_extension(7, custom_file)
    assert custom_file.read_text() == "7"


def test_add_extension_swaps_reader_for_every_datatype_using_it(bids_root):
    dm = DatasetManager(bids_root)

    dm.add_extension(".nii.gz", reader=lambda p: "swapped")

    assert dm.load("01", "T1w") == "swapped"
    assert ".nii.gz" in dm.ext_writers  # writer untouched


def test_save_by_extension_unknown_extension_raises_value_error(bids_root, tmp_path):
    dm = DatasetManager(bids_root)

    with pytest.raises(ValueError):
        dm.save_by_extension("data", tmp_path / "data.mystery")


# --- set_reader / set_writer / describe --------------------------------------


def test_set_reader_swaps_one_datatype_only(bids_root):
    dm = DatasetManager(bids_root)
    dm.add_datatype("T1w_copy", "raw", dm.bids_namer("T1w"))

    dm.set_reader("T1w", lambda p: "custom")

    assert dm.load("01", "T1w") == "custom"
    assert dm.load("01", "T1w_copy").shape == (2, 3, 4)


def test_set_reader_none_restores_extension_dispatch(bids_root):
    dm = DatasetManager(bids_root)
    dm.set_reader("T1w", lambda p: "custom")

    dm.set_reader("T1w", None)

    assert dm.load("01", "T1w").shape == (2, 3, 4)


def test_set_writer(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    _text_root(tmp_path, dm)
    dm.add_datatype("custom", "custom", _custom_namer, _read_text)
    written = {}

    dm.set_writer("custom", lambda data, path: written.update(data=data, path=path))
    path = dm.save("bye", "01", "custom", overwrite=True)

    assert written == {"data": "bye", "path": path}


@pytest.mark.parametrize("method", ["set_reader", "set_writer"])
def test_set_unknown_key_raises_key_error(bids_root, method):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError, match=r"T1w"):
        getattr(dm, method)("bogus", lambda *a: None)


def test_describe(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    _text_root(tmp_path, dm)
    dm.add_datatype("custom", "custom", _custom_namer, _read_text, inputs=("hemi",))

    info = dm.describe("custom")

    assert info["root_key"] == "custom"
    assert info["root"] == tmp_path / "custom"
    assert info["namer"] == "_custom_namer"
    assert info["reader"] == "_read_text"
    assert info["writer"] == "by extension"
    assert info["inputs"] == ("hemi",)
    assert dm.describe("T1w")["namer"] == "bids_namer('T1w')"


# --- get_path / load / save --------------------------------------------------


def test_get_path_does_not_require_file_to_exist(bids_root):
    dm = DatasetManager(bids_root)

    path = dm.get_path("01", "T1w", session="new")

    assert not path.exists()
    assert path.name == "sub-01_ses-new_T1w.nii.gz"


def test_load_missing_file_raises_file_not_found(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(FileNotFoundError, match="sub-01_ses-new_T1w"):
        dm.load("01", "T1w", session="new")


def test_save_t1w_into_raw_round_trip(bids_root):
    dm = DatasetManager(bids_root)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    path = dm.save(nib.Nifti1Image(data, np.eye(4)), "01", "T1w", session="new")

    assert path == bids_root / "sub-01" / "ses-new" / "anat" / "sub-01_ses-new_T1w.nii.gz"
    np.testing.assert_array_equal(dm.load("01", "T1w", session="new").get_fdata(), data)


def test_save_existing_file_requires_overwrite(bids_root):
    dm = DatasetManager(bids_root)
    img = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.float32), np.eye(4))

    with pytest.raises(FileExistsError):
        dm.save(img, "01", "T1w")

    dm.save(img, "01", "T1w", overwrite=True)
    np.testing.assert_array_equal(dm.load("01", "T1w").get_fdata(), 1.0)


def test_save_read_only_datatype_raises_type_error(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    _text_root(tmp_path, dm)
    dm.add_datatype("custom", "custom", _custom_namer, _read_text)  # .txt: no ext writer

    with pytest.raises(TypeError, match="read-only"):
        dm.save("bye", "01", "custom", overwrite=True)


def test_save_custom_writer_creates_parent_dirs(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    dm.add_root("out", tmp_path / "out")

    def namer(root, subject, **inputs):
        return root / f"sub-{subject}" / "deep" / "note.txt"

    dm.add_datatype("note", "out", namer, _read_text, _write_text)
    path = dm.save("hi", "01", "note")

    assert path.read_text() == "hi"
    assert dm.load("01", "note") == "hi"


def test_exists_override_used_by_load_and_save(bids_root, tmp_path):
    """A namer returning a multi-file stem relies on a custom `exists`."""
    dm = DatasetManager(bids_root)
    dm.add_root("out", tmp_path)

    def namer(root, subject, **inputs):
        return root / f"sub-{subject}_pair"

    def write_pair(data, stem):
        for hemi in ("lh", "rh"):
            stem.with_name(f"{stem.name}-{hemi}.txt").write_text(data)

    def read_pair(stem):
        return stem.with_name(f"{stem.name}-lh.txt").read_text()

    def pair_exists(stem):
        return stem.with_name(f"{stem.name}-lh.txt").exists()

    dm.add_datatype("pair", "out", namer, read_pair, write_pair, exists=pair_exists)

    dm.save("x", "01", "pair")
    assert dm.load("01", "pair") == "x"
    with pytest.raises(FileExistsError):
        dm.save("y", "01", "pair")


# --- refs / load_ref -------------------------------------------------------


def test_add_reference_and_load_ref(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    ref_file = tmp_path / "template.nii.gz"
    data = np.ones((2, 3, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), ref_file)

    dm.add_reference("MY_REF", ref_file)
    loaded = dm.load_ref("MY_REF")

    np.testing.assert_array_equal(loaded.get_fdata(), data)


def test_load_ref_unknown_key_raises_key_error(bids_root):
    dm = DatasetManager(bids_root)
    dm.add_reference("MY_REF", "somewhere.nii.gz")

    with pytest.raises(KeyError, match="MY_REF"):
        dm.load_ref("NOT_A_REF")


def test_add_reference_with_own_reader(bids_root, tmp_path):
    """A reference with no extension reader (.txt) loads through its own reader."""
    dm = DatasetManager(bids_root)
    lut = tmp_path / "FreeSurferColorLUT.txt"
    lut.write_text("0 Unknown\n2 Left-Cerebral-White-Matter\n")

    dm.add_reference("LUT_REF", lut, reader=lambda p: p.read_text().splitlines())

    assert dm.load_ref("LUT_REF") == ["0 Unknown", "2 Left-Cerebral-White-Matter"]


def test_ref_reader_only_affects_that_reference(bids_root, tmp_path):
    dm = DatasetManager(bids_root)
    for name in ("a", "b"):
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), tmp_path / f"{name}.nii.gz")

    dm.add_reference("A_REF", tmp_path / "a.nii.gz", reader=lambda p: "custom")
    dm.add_reference("B_REF", tmp_path / "b.nii.gz")

    assert dm.load_ref("A_REF") == "custom"
    assert dm.load_ref("B_REF").shape == (2, 2, 2)
    assert dm.load("01", "T1w").shape == (2, 3, 4)  # datatypes untouched


def test_set_ref_reader_on_config_reference(bids_root, tmp_path):
    ref_file = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), ref_file)
    dm = DatasetManager(bids_root, config={"MRI_REF": ref_file})

    dm.set_ref_reader("MRI_REF", lambda p: ("custom", p))
    assert dm.load_ref("MRI_REF") == ("custom", ref_file)

    dm.set_ref_reader("MRI_REF", None)
    assert dm.load_ref("MRI_REF").shape == (2, 2, 2)


def test_set_ref_reader_unknown_key_raises_key_error(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError):
        dm.set_ref_reader("NOT_A_REF", lambda p: p)


def test_re_adding_reference_without_reader_clears_old_reader(bids_root, tmp_path):
    ref_file = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), ref_file)
    dm = DatasetManager(bids_root)

    dm.add_reference("MRI_REF", ref_file, reader=lambda p: "custom")
    dm.add_reference("MRI_REF", ref_file)

    assert dm.load_ref("MRI_REF").shape == (2, 2, 2)


# --- add_derivatives --------------------------------------------------------


def test_add_derivatives(bids_root):
    deriv_root = bids_root / "derivatives" / "mytool"
    deriv_anat = deriv_root / "sub-01" / "anat"
    deriv_anat.mkdir(parents=True)
    (deriv_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "MyTool derivatives",
                "BIDSVersion": "1.8.0",
                "GeneratedBy": [{"Name": "mytool"}],
            }
        )
    )
    img = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
    nib.save(img, deriv_anat / "sub-01_desc-processed_T1w.nii.gz")

    dm = DatasetManager(bids_root)
    dm.add_derivatives("mytool", deriv_root)

    assert dm.roots["mytool"] == deriv_root
    assert "mytool" in dm.layout.derivatives


def test_read_only_writer_refuses_save_even_with_extension_writer(bids_root):
    dm = DatasetManager(bids_root)
    dm.set_writer("T1w", Datatype.READ_ONLY)

    with pytest.raises(TypeError, match="read-only"):
        dm.save(object(), "01", "T1w", overwrite=True)


def test_read_only_is_a_sentinel_not_a_field(bids_root):
    from dataclasses import fields

    dt = Datatype(lambda root, subject: "p", writer=Datatype.READ_ONLY)

    assert "READ_ONLY" not in {f.name for f in fields(Datatype)}
    assert dt.READ_ONLY is Datatype.READ_ONLY  # same object via instance or class
    with pytest.raises(TypeError, match="read-only"):
        dt.writer(None, "some/path")


# --- raw namer is strict (regression: entities used to be silently dropped) --


@pytest.mark.parametrize("entities", [{"desc": "brain"}, {"space": "MNI152"}, {"myentity": "x"}])
def test_raw_namer_rejects_entities_outside_raw_patterns(bids_root, entities):
    dm = DatasetManager(bids_root)

    with pytest.raises(ValueError, match="derivative_namer"):
        dm.get_path("01", "T1w", **entities)


def test_save_with_derivative_entity_does_not_overwrite_raw_t1w(bids_root):
    dm = DatasetManager(bids_root)
    raw = bids_root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    before = raw.read_bytes()
    mask = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.float32), np.eye(4))

    with pytest.raises(ValueError):
        dm.save(mask, "01", "T1w", desc="brain", overwrite=True)

    assert raw.read_bytes() == before


def test_raw_namer_still_accepts_raw_entities_and_sidecar(bids_root):
    dm = DatasetManager(bids_root)

    path = dm.get_path("01", "T1w", session="a", acquisition="mprage", run=2, extension=".json")

    assert path.name == "sub-01_ses-a_acq-mprage_run-2_T1w.json"


# --- raw modalities found through the BIDS index (add_raw) -----------------


def test_add_raw_finds_modality_bids_does_not_define(raw_root):
    dm = DatasetManager(raw_root)
    key = dm.add_raw("bar", "bar", reader=lambda p: p.name)

    assert key == "bar"
    assert dm.load("01", "bar", acquisition="y") == "sub-01_task-x_acq-y_bar.xyz"


def test_add_raw_key_needs_not_be_the_suffix(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("bar-name", "bar", reader=lambda p: p.name)

    assert dm.load("01", "bar-name") == "sub-01_task-x_acq-y_bar.xyz"


@pytest.mark.parametrize("run", [1, "01"])
def test_add_raw_matches_run_whatever_the_padding(raw_root, run):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg", extensions=(".fif", ".ds"))

    path = dm.get_path("01", "meg", session="a", task="x", run=run)

    assert path.name == "sub-01_ses-a_task-x_run-01_meg.fif"


def test_add_raw_returns_directory_recording(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg", extensions=(".fif", ".ds"))

    path = dm.get_path("01", "meg", session="a", task="y")

    assert path.is_dir()
    assert path.name == "sub-01_ses-a_task-y_meg.ds"


def test_add_raw_ignores_directory_with_other_extension(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg", extensions=(".fif",))

    with pytest.raises(FileNotFoundError):
        dm.get_path("01", "meg", session="a", task="y")


def test_add_raw_extensions_skip_companion_files(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("eeg", "eeg", extensions=(".vhdr", ".edf"))

    assert dm.get_path("01", "eeg", task="x").name == "sub-01_task-x_eeg.vhdr"


def test_add_raw_extension_argument_takes_precedence(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("eeg", "eeg", extensions=(".vhdr",))

    assert dm.get_path("01", "eeg", task="x", extension=".json").suffix == ".json"


def test_add_raw_several_matches_raise_and_list_candidates(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("eeg", "eeg")  # any extension: .vhdr, .eeg, .vmrk and .json all match

    with pytest.raises(ValueError, match=r"Several raw .eeg. files.*\.vmrk"):
        dm.get_path("01", "eeg", task="x")


def test_add_raw_missing_file_raises_file_not_found(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg", extensions=(".fif",))

    with pytest.raises(FileNotFoundError, match="task"):
        dm.load("01", "meg", session="a", task="none")


def test_add_raw_unknown_entity_raises(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("bar", "bar")

    with pytest.raises(ValueError, match="bogus"):
        dm.get_path("01", "bar", bogus="x")


def test_add_raw_filters_apply_to_every_lookup(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("bar", "bar", filters={"acquisition": "z"})

    with pytest.raises(FileNotFoundError, match="'acquisition': 'z'"):
        dm.get_path("01", "bar")


def test_add_raw_is_read_only(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("bar", "bar")

    with pytest.raises(TypeError, match="read-only"):
        dm.save(object(), "01", "bar", overwrite=True)


def test_add_raw_describe(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("bar", "bar")

    info = dm.describe("bar")

    assert info["root_key"] == "raw"
    assert info["root"] == raw_root
    assert info["namer"] == "raw_lookup('bar')"
    assert info["writer"] == "READ_ONLY"


# --- sidecars of raw recordings (add_raw sidecar_of=) --------------------------


def _meg_with_sidecar(raw_root, **sidecar):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg", extensions=(".fif", ".ds"))
    dm.add_raw("meg-channels", "channels", lambda p: p.name, extensions=(".tsv",),
               sidecar_of="meg", **sidecar)
    return dm


def test_sidecar_inherited_from_session_level(raw_root):
    channels = raw_root / "sub-01" / "ses-a" / "sub-01_ses-a_channels.tsv"
    channels.touch()
    dm = _meg_with_sidecar(raw_root)

    assert dm.get_path("01", "meg-channels", session="a", task="x", run=1) == channels
    assert dm.load("01", "meg-channels", session="a", task="x", run=1) == channels.name


def test_sidecar_shared_by_several_recordings(raw_root):
    channels = raw_root / "sub-01" / "ses-a" / "sub-01_ses-a_channels.tsv"
    channels.touch()
    dm = _meg_with_sidecar(raw_root)

    # session="a" matches both the .fif run and the .ds recording
    assert dm.get_path("01", "meg-channels", session="a") == channels


def test_sidecar_differing_between_recordings_raises(raw_root):
    meg = raw_root / "sub-01" / "ses-a" / "meg"
    (meg / "sub-01_ses-a_task-x_channels.tsv").touch()
    (meg / "sub-01_ses-a_task-y_channels.tsv").touch()
    dm = _meg_with_sidecar(raw_root)

    with pytest.raises(ValueError, match="different 'channels' sidecars"):
        dm.get_path("01", "meg-channels", session="a")
    assert dm.get_path("01", "meg-channels", session="a", task="y").name == (
        "sub-01_ses-a_task-y_channels.tsv"
    )


def test_sidecar_missing_raises_file_not_found(raw_root):
    dm = _meg_with_sidecar(raw_root)

    with pytest.raises(FileNotFoundError, match="channels"):
        dm.load("01", "meg-channels", session="a", task="x")


def test_sidecar_of_must_be_a_raw_key(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("t1-sidecar", "T1w", extensions=(".json",), sidecar_of="T1w")

    with pytest.raises(TypeError, match="add_raw"):
        dm.get_path("01", "t1-sidecar")


def test_sidecar_describe(raw_root):
    dm = _meg_with_sidecar(raw_root)

    assert dm.describe("meg-channels")["namer"] == "raw_sidecar('channels' of 'meg')"
    assert dm.describe("meg-channels")["writer"] == "READ_ONLY"


def test_add_raw_any_extension_returns_directory_recording(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg")

    assert dm.get_path("01", "meg", session="a", task="y").name == "sub-01_ses-a_task-y_meg.ds"


def test_directory_lookup_stops_at_raw_root(tmp_path):
    root = tmp_path / "study_meg.ds"  # the raw root itself looks like a recording
    meg = root / "sub-01" / "meg"
    meg.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.8.0"}')
    (meg / "sub-01_task-x_meg.fif").touch()
    dm = DatasetManager(root)
    dm.add_raw("meg", "meg")

    assert dm.get_path("01", "meg", task="x").name == "sub-01_task-x_meg.fif"


def test_sidecar_extension_argument_picks_sidecar_extension(raw_root):
    session = raw_root / "sub-01" / "ses-a"
    (session / "sub-01_ses-a_channels.tsv").touch()
    json_sidecar = session / "sub-01_ses-a_channels.json"
    json_sidecar.write_text("{}")  # pybids parses json sidecars
    dm = _meg_with_sidecar(raw_root)

    assert dm.get_path("01", "meg-channels", session="a", task="x").suffix == ".tsv"
    assert dm.get_path("01", "meg-channels", session="a", task="x", extension=".json") == (
        json_sidecar
    )


def test_sidecar_missing_recording_names_recording_query(raw_root):
    (raw_root / "sub-01" / "ses-a" / "sub-01_ses-a_channels.tsv").touch()
    dm = _meg_with_sidecar(raw_root)

    with pytest.raises(FileNotFoundError, match=r"'suffix': 'meg'.*'task': 'none'"):
        dm.get_path("01", "meg-channels", session="a", task="none")


def test_sidecar_of_unregistered_key_raises_key_error(raw_root):
    dm = DatasetManager(raw_root)
    dm.add_raw("meg-channels", "channels", extensions=(".tsv",), sidecar_of="nope")

    with pytest.raises(KeyError, match="nope"):
        dm.get_path("01", "meg-channels")


def test_sidecar_follows_re_registered_recording_key(raw_root):
    meg = raw_root / "sub-01" / "ses-a" / "meg"
    (meg / "sub-01_ses-a_task-x_channels.tsv").touch()
    ds_channels = meg / "sub-01_ses-a_task-y_channels.tsv"
    ds_channels.touch()
    dm = DatasetManager(raw_root)
    dm.add_raw("meg", "meg", extensions=(".fif",))
    dm.add_raw("meg-channels", "channels", extensions=(".tsv",), sidecar_of="meg")
    with pytest.raises(FileNotFoundError):  # task y is only a .ds recording
        dm.get_path("01", "meg-channels", session="a", task="y")

    dm.add_raw("meg", "meg", extensions=(".ds",))

    assert dm.get_path("01", "meg-channels", session="a", task="y") == ds_channels
