"""Tests for fascicle.extensions.mindmap: mindmap-bem's BEM surfaces."""

import nibabel as nib
import numpy as np
import pytest

from fascicle import DatasetManager
from fascicle.extensions import mindmap

# key -> file written by mindmap-bem (MNE's names)
FILES = {"iskull": "inner_skull.surf", "oskull": "outer_skull.surf", "skin": "outer_skin.surf"}
COORDS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
FACES = np.array([[0, 1, 2]], dtype=np.int32)


@pytest.fixture
def mindmap_root(tmp_path):
    """An MNE subjects dir as mindmap-bem writes it: sub-01/bem/*.surf."""
    root = tmp_path / "mindmap"
    bem = root / "sub-01" / "bem"
    bem.mkdir(parents=True)
    for scale, file in enumerate(FILES.values(), start=1):
        nib.freesurfer.io.write_geometry(bem / file, COORDS * scale, FACES)
    return root


@pytest.fixture
def dm(bids_root, mindmap_root):
    manager = DatasetManager(bids_root)
    mindmap.register(manager, root_path=mindmap_root)
    return manager


def test_register_keys(dm):
    assert {f"mm-{name}" for name in FILES} <= set(dm.registry)
    assert dm.describe("mm-iskull")["root_key"] == "mm"


@pytest.mark.parametrize("name, file", FILES.items())
def test_paths(dm, mindmap_root, name, file):
    assert dm.get_path("01", f"mm-{name}") == mindmap_root / "sub-01" / "bem" / file


@pytest.mark.parametrize("subject", ["01", "sub-01"])
def test_load_with_either_subject_form(dm, subject):
    coords, faces = dm.load(subject, "mm-skin")

    np.testing.assert_array_equal(coords, COORDS * 3)
    np.testing.assert_array_equal(faces, FACES)


def test_save_round_trip_keeps_volume_info(dm, mindmap_root, volume_info):
    path = dm.save((COORDS * 7, FACES, volume_info), "02", "mm-iskull")
    *_, written = nib.freesurfer.io.read_geometry(path, read_metadata=True)

    assert path == mindmap_root / "sub-02" / "bem" / "inner_skull.surf"
    np.testing.assert_array_equal(dm.load("02", "mm-iskull")[0], COORDS * 7)
    np.testing.assert_array_equal(written["cras"], [1.0, 2.0, 3.0])


def test_save_refuses_to_overwrite(dm):
    with pytest.raises(FileExistsError):
        dm.save((COORDS, FACES), "01", "mm-iskull")


def test_custom_root_key_and_namespace(bids_root, mindmap_root):
    manager = DatasetManager(bids_root)
    mindmap.register(manager, root_key="mm-shared", root_path=mindmap_root, namespace="shared")

    assert "shared-oskull" in manager.registry
    assert manager.describe("shared-oskull")["root_key"] == "mm-shared"


def test_missing_root_raises(bids_root):
    manager = DatasetManager(bids_root)

    with pytest.raises(KeyError, match="'mm'"):
        mindmap.register(manager)
