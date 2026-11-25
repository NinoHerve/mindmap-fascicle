"""Tests for fascicle.utils: paths, extensions, find_sidecar and find_padded."""

import pytest

from fascicle.utils import (
    _filename_entities,
    build_bids_path,
    check_file,
    find_padded,
    find_sidecar,
    parse_extension,
)


class _FakeLayout:
    """Minimal stand-in for a pybids layout, just exercising build_path."""

    def __init__(self, path=None):
        self._path = path

    def build_path(self, entities, validate=False, strict=False):
        return self._path


# --- parse_extension -------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("sub-01_T1w.nii.gz", ".nii.gz"),
        ("sub-01_T1w.nii", ".nii"),
        ("sub-01_desc-foo.mgz", ".mgz"),
        ("sub-01_task-rest_physio.tsv.gz", ".tsv.gz"),
        ("sub-01_task-rest_events.tsv", ".tsv"),
        ("archive.tar.gz", ".gz"),
        ("noext", ""),
    ],
)
def test_parse_extension(filename, expected):
    assert parse_extension(filename) == expected


# --- check_file --------------------------------------------------------


def test_check_file_missing_raises(tmp_path):
    missing = tmp_path / "nope.nii.gz"
    with pytest.raises(FileNotFoundError):
        check_file(missing)


def test_check_file_returns_path_by_default(tmp_path):
    f = tmp_path / "sub-01_T1w.nii.gz"
    f.write_text("data")

    result = check_file(f)

    assert result == f
    assert isinstance(result, type(f))


def test_check_file_instance_str(tmp_path):
    f = tmp_path / "sub-01_T1w.nii.gz"
    f.write_text("data")

    result = check_file(f, instance="str")

    assert result == str(f)


@pytest.mark.parametrize("extension", [".nii.gz", "nii.gz", [".mgz", ".nii.gz"]])
def test_check_file_extension_match(tmp_path, extension):
    f = tmp_path / "sub-01_T1w.nii.gz"
    f.write_text("data")

    assert check_file(f, extension=extension) == f


def test_check_file_extension_mismatch_raises(tmp_path):
    f = tmp_path / "sub-01_T1w.nii.gz"
    f.write_text("data")

    with pytest.raises(ValueError):
        check_file(f, extension=".mgz")


# --- build_bids_path ---------------------------------------------------


def test_build_bids_path_success():
    layout = _FakeLayout(path="/bids/sub-01/anat/sub-01_T1w.nii.gz")

    result = build_bids_path(layout, subject="01", suffix="T1w")

    assert result == "/bids/sub-01/anat/sub-01_T1w.nii.gz"


def test_build_bids_path_passes_kwargs_as_entities():
    seen = {}

    class RecordingLayout:
        def build_path(self, entities, validate=False, strict=False):
            seen.update(entities, strict=strict)
            return "some/path"

    build_bids_path(RecordingLayout(), subject="01", suffix="bold", task="rest", run=1)

    assert seen == {"subject": "01", "suffix": "bold", "task": "rest", "run": 1, "strict": True}


def test_build_bids_path_unresolvable_raises():
    layout = _FakeLayout(path=None)

    with pytest.raises(ValueError):
        build_bids_path(layout, subject="01", suffix="T1w")


# --- find_sidecar (BIDS inheritance) ----------------------------------------


@pytest.fixture
def dataset(tmp_path):
    """A recording two levels below the dataset root."""
    root = tmp_path / "ds"
    meg = root / "sub-01" / "ses-a" / "meg"
    meg.mkdir(parents=True)
    (root / "dataset_description.json").write_text("{}")
    recording = meg / "sub-01_ses-a_task-x_run-01_meg.fif"
    recording.touch()
    return root, recording


def test_find_sidecar_inherits_from_session_level(dataset):
    root, recording = dataset
    channels = root / "sub-01" / "ses-a" / "sub-01_ses-a_task-x_channels.tsv"
    channels.touch()

    assert find_sidecar(recording, "channels", ".tsv") == channels


def test_find_sidecar_from_dataset_root(dataset):
    root, recording = dataset
    sidecar = root / "task-x_meg.json"
    sidecar.touch()

    assert find_sidecar(recording, "meg", ".json") == sidecar


def test_find_sidecar_nearest_level_wins(dataset):
    root, recording = dataset
    (root / "task-x_channels.tsv").touch()
    near = recording.parent / "sub-01_ses-a_channels.tsv"
    near.touch()

    assert find_sidecar(recording, "channels", ".tsv") == near


def test_find_sidecar_most_specific_wins_within_a_level(dataset):
    _, recording = dataset
    (recording.parent / "sub-01_ses-a_channels.tsv").touch()
    specific = recording.parent / "sub-01_ses-a_task-x_run-01_channels.tsv"
    specific.touch()

    assert find_sidecar(recording, "channels", ".tsv") == specific


def test_find_sidecar_ignores_sidecars_of_other_recordings(dataset):
    _, recording = dataset
    (recording.parent / "sub-01_ses-a_task-y_channels.tsv").touch()
    (recording.parent / "sub-01_ses-a_task-x_run-02_channels.tsv").touch()

    with pytest.raises(FileNotFoundError, match="channels"):
        find_sidecar(recording, "channels", ".tsv")


def test_find_sidecar_equally_specific_candidates_raise(dataset):
    _, recording = dataset
    (recording.parent / "sub-01_ses-a_task-x_channels.tsv").touch()
    (recording.parent / "sub-01_ses-a_run-01_channels.tsv").touch()

    with pytest.raises(ValueError, match="Several"):
        find_sidecar(recording, "channels", ".tsv")


def test_find_sidecar_stops_at_dataset_root(dataset, tmp_path):
    _, recording = dataset
    (tmp_path / "task-x_meg.json").touch()  # above the dataset root

    with pytest.raises(FileNotFoundError):
        find_sidecar(recording, "meg", ".json")


def test_find_sidecar_matches_bare_dataset_level_name(dataset):
    root, recording = dataset
    sidecar = root / "meg.json"
    sidecar.touch()

    assert find_sidecar(recording, "meg", ".json") == sidecar


def test_find_sidecar_for_path_whose_folder_does_not_exist(dataset):
    root, _ = dataset
    channels = root / "sub-01" / "ses-a" / "sub-01_ses-a_task-x_channels.tsv"
    channels.touch()
    missing = root / "sub-01" / "ses-a" / "eeg" / "sub-01_ses-a_task-x_channels.tsv"

    assert find_sidecar(missing, "channels", ".tsv") == channels


def test_find_sidecar_compares_index_entities_as_numbers(dataset):
    root, recording = dataset  # ..._run-01_meg.fif
    sidecar = root / "run-1_meg.json"
    sidecar.touch()

    assert find_sidecar(recording, "meg", ".json") == sidecar


def test_find_sidecar_split_sidecar_applies_to_its_split_only(dataset):
    _, recording = dataset
    split = recording.with_name("sub-01_ses-a_task-x_split-01_meg.fif")
    split.touch()
    (recording.parent / "sub-01_ses-a_task-x_split-02_channels.tsv").touch()
    own = recording.parent / "sub-01_ses-a_task-x_split-1_channels.tsv"
    own.touch()

    assert find_sidecar(split, "channels", ".tsv") == own


def test_find_sidecar_for_directory_recording(dataset):
    _, recording = dataset
    ds = recording.parent / "sub-01_ses-a_task-y_meg.ds"
    ds.mkdir()
    channels = recording.parent / "sub-01_ses-a_task-y_channels.tsv"
    channels.touch()

    assert find_sidecar(ds, "channels", ".tsv") == channels


def test_find_sidecar_without_description_searches_up_to_dataset_folder(dataset):
    root, recording = dataset
    (root / "dataset_description.json").unlink()
    sidecar = root / "task-x_meg.json"
    sidecar.touch()

    assert find_sidecar(recording, "meg", ".json") == sidecar


def test_find_sidecar_without_description_stops_above_subject(dataset, tmp_path):
    root, recording = dataset
    (root / "dataset_description.json").unlink()
    (tmp_path / "task-x_meg.json").touch()  # outside the dataset folder

    with pytest.raises(FileNotFoundError):
        find_sidecar(recording, "meg", ".json")


def test_find_sidecar_outside_bids_searches_own_folder_only(tmp_path):
    folder = tmp_path / "loose" / "files"
    folder.mkdir(parents=True)
    recording = folder / "sub-01_task-x_meg.fif"
    recording.touch()
    (tmp_path / "loose" / "task-x_meg.json").touch()  # a level up: not searched
    here = folder / "sub-01_task-x_channels.tsv"
    here.touch()

    assert find_sidecar(recording, "channels", ".tsv") == here
    with pytest.raises(FileNotFoundError):
        find_sidecar(recording, "meg", ".json")


def test_find_sidecar_accepts_several_extensions(dataset):
    _, recording = dataset
    sidecar = recording.parent / "sub-01_ses-a_task-x_run-01_channels.json"
    sidecar.touch()

    assert find_sidecar(recording, "channels", (".tsv", ".json")) == sidecar


def test_find_sidecar_skips_folder_named_like_a_sidecar(dataset):
    _, recording = dataset
    (recording.parent / "sub-01_ses-a_task-x_channels.tsv").mkdir()

    with pytest.raises(FileNotFoundError):
        find_sidecar(recording, "channels", ".tsv")


# --- zero-padding of index entities (find_padded) ------------------------------


def test_filename_entities_reads_index_values_as_numbers():
    entities, suffix, extension = _filename_entities("sub-01_ses-a_run-001_echo-2_T1w.nii.gz")

    assert entities == {"sub": "01", "ses": "a", "run": 1, "echo": 2}
    assert (suffix, extension) == ("T1w", ".nii.gz")


def _anat(tmp_path, *names):
    anat = tmp_path / "sub-01" / "anat"
    anat.mkdir(parents=True)
    for name in names:
        (anat / name).touch()
    return anat


def test_find_padded_matches_other_padding(tmp_path):
    anat = _anat(tmp_path, "sub-01_run-01_T1w.nii.gz")

    assert find_padded(anat / "sub-01_run-1_T1w.nii.gz") == [anat / "sub-01_run-01_T1w.nii.gz"]


def test_find_padded_ignores_sidecar_with_other_extension(tmp_path):
    anat = _anat(tmp_path, "sub-01_run-01_T1w.json")

    assert find_padded(anat / "sub-01_run-1_T1w.nii.gz") == []


def test_find_padded_requires_the_same_entities(tmp_path):
    anat = _anat(tmp_path, "sub-01_acq-x_run-01_T1w.nii.gz")

    assert find_padded(anat / "sub-01_run-1_T1w.nii.gz") == []


def test_find_padded_does_not_pad_labels(tmp_path):
    anat = _anat(tmp_path, "sub-01_acq-01_T1w.nii.gz")

    assert find_padded(anat / "sub-01_acq-1_T1w.nii.gz") == []


def test_find_padded_missing_folder(tmp_path):
    assert find_padded(tmp_path / "nope" / "sub-01_run-1_T1w.nii.gz") == []


def test_subject_label_strips_leading_prefix_only():
    from fascicle.utils import subject_label

    assert subject_label("sub-01") == subject_label("01") == "01"
    assert subject_label("01sub-x") == "01sub-x"
