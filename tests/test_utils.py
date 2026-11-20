"""Tests for fascicle.utils: parse_extension, check_file, build_bids_path."""

import pytest

from fascicle.utils import build_bids_path, check_file, parse_extension


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
