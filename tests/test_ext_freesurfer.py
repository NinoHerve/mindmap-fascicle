"""Tests for fascicle.extensions.freesurfer."""

import nibabel as nib
import numpy as np
import pytest

from fascicle import DatasetManager
from fascicle.extensions import freesurfer


# --- build_fs_path (operates on bare, unprefixed datatype names) ----------


@pytest.mark.parametrize(
    "datatype, hemi, expected_suffix",
    [
        ("t1", "lh", "sub-01/mri/T1.mgz"),
        ("brain", "lh", "sub-01/mri/brain.mgz"),
        ("orig", "lh", "sub-01/mri/orig.mgz"),
        ("aseg", "lh", "sub-01/mri/aseg.mgz"),
        ("white", "lh", "sub-01/surf/lh.white"),
        ("white", "rh", "sub-01/surf/rh.white"),
        ("pial", "rh", "sub-01/surf/rh.pial"),
        ("inflated", "lh", "sub-01/surf/lh.inflated"),
        ("sphere", "lh", "sub-01/surf/lh.sphere"),
        ("area", "lh", "sub-01/surf/lh.area"),
        ("curv", "lh", "sub-01/surf/lh.curv"),
        ("thickness", "rh", "sub-01/surf/rh.thickness"),
        ("sulc", "rh", "sub-01/surf/rh.sulc"),
        ("cortex", "rh", "sub-01/label/rh.cortex.label"),
    ],
)
def test_build_fs_path_datatype_and_hemi(datatype, hemi, expected_suffix):
    path = freesurfer.build_fs_path("/fs", "01", datatype, hemi=hemi)
    assert str(path) == f"/fs/{expected_suffix}"


def test_build_fs_path_strips_sub_prefix():
    with_prefix = freesurfer.build_fs_path("/fs", "sub-01", "t1")
    without_prefix = freesurfer.build_fs_path("/fs", "01", "t1")
    assert with_prefix == without_prefix


def test_build_fs_path_unknown_datatype_raises():
    with pytest.raises(ValueError):
        freesurfer.build_fs_path("/fs", "01", "bogus")


def test_build_fs_path_excludes_bem():
    with pytest.raises(ValueError):
        freesurfer.build_fs_path("/fs", "01", "bem")


# --- register (registry keys are prefixed f"{root_key}-<name>") ----------


def test_register_missing_root_raises(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError):
        freesurfer.register(dm)


def test_register_with_root_path_adds_root(bids_root, fs_root):
    dm = DatasetManager(bids_root)

    freesurfer.register(dm, root_key="fs", root_path=fs_root)

    assert dm.roots["fs"] == fs_root


def test_register_with_existing_root_key(bids_root, fs_root):
    dm = DatasetManager(bids_root)
    dm.add_root("fs", fs_root)

    freesurfer.register(dm, root_key="fs")

    assert "fs-t1" in dm.registry


def test_register_keys_use_custom_root_key_as_prefix(bids_root, fs_root):
    """Registering twice under different root_keys must not collide."""
    dm = DatasetManager(bids_root)

    freesurfer.register(dm, root_key="fs", root_path=fs_root)
    freesurfer.register(dm, root_key="fs_run2", root_path=fs_root)

    assert "fs-t1" in dm.registry
    assert "fs_run2-t1" in dm.registry
    assert dm.load("01", "fs-t1").shape == dm.load("01", "fs_run2-t1").shape


def test_register_namespace_decoupled_from_root_key(bids_root, fs_root):
    """namespace= lets the key prefix differ from the root's own name."""
    dm = DatasetManager(bids_root)

    freesurfer.register(dm, root_key="my_custom_root", root_path=fs_root, namespace="fs")

    assert "fs-t1" in dm.registry
    assert "my_custom_root-t1" not in dm.registry
    assert dm.load("01", "fs-t1").shape == (5, 5, 5)


# --- loading -----------------------------------------------------------


@pytest.fixture
def dm(bids_root, fs_root):
    manager = DatasetManager(bids_root)
    freesurfer.register(manager, root_key="fs", root_path=fs_root)
    return manager


def test_load_t1_via_extension_registry(dm):
    img = dm.load("01", "fs-t1")
    assert img.shape == (5, 5, 5)


def test_load_aseg_via_extension_registry(dm):
    img = dm.load("01", "fs-aseg")
    assert img.shape == (5, 5, 5)


@pytest.mark.parametrize("hemi", ["lh", "rh"])
@pytest.mark.parametrize("name", ["white", "pial", "sphere"])
def test_load_surface_per_hemi(dm, name, hemi):
    coords, faces = dm.load("01", f"fs-{name}", hemi=hemi)
    assert coords.shape == (3, 3)
    assert faces.shape == (1, 3)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("area", [1.0, 2.0, 3.0]),
        ("curv", [0.1, 0.2, 0.3]),
        ("thickness", [2.0, 2.5, 3.0]),
        ("sulc", [-1.0, 0.0, 1.0]),
    ],
)
@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_load_morph_per_hemi(dm, name, expected, hemi):
    np.testing.assert_array_almost_equal(dm.load("01", f"fs-{name}", hemi=hemi), expected)


def test_load_cortex_label(dm):
    np.testing.assert_array_equal(dm.load("01", "fs-cortex", hemi="rh"), [0, 1])


def test_hemi_datatype_without_hemi_raises_value_error(dm):
    with pytest.raises(ValueError, match="hemi"):
        dm.load("01", "fs-pial")


def test_hemi_datatypes_declare_hemi_input(dm):
    assert dm.describe("fs-pial")["inputs"] == ("hemi",)
    assert dm.describe("fs-t1")["inputs"] == ()


def test_missing_hemisphere_raises_file_not_found_error(dm, fs_root):
    (fs_root / "sub-01" / "surf" / "rh.pial").unlink()

    assert dm.load("01", "fs-pial", hemi="lh")
    with pytest.raises(FileNotFoundError, match="rh.pial"):
        dm.load("01", "fs-pial", hemi="rh")


def test_load_ras2tkras(dm, fs_root):
    transform = dm.load("01", "fs-ras2tkras")

    t1_path = fs_root / "sub-01" / "mri" / "T1.mgz"
    header = nib.load(t1_path).header
    expected = header.get_vox2ras_tkr() @ np.linalg.inv(header.get_vox2ras())

    assert transform.shape == (4, 4)
    np.testing.assert_array_almost_equal(transform, expected)


# --- saving ------------------------------------------------------------


def test_save_t1_round_trip(dm, fs_root):
    data = np.full((5, 5, 5), 3.0, dtype=np.float32)

    path = dm.save(nib.MGHImage(data, np.eye(4)), "01", "fs-t1", overwrite=True)

    assert path == fs_root / "sub-01" / "mri" / "T1.mgz"
    np.testing.assert_array_equal(dm.load("01", "fs-t1").get_fdata(), data)


def test_save_surface_round_trip_into_new_subject(dm, fs_root):
    coords = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    path = dm.save((coords, faces), "02", "fs-pial", hemi="lh")
    loaded_coords, loaded_faces = dm.load("02", "fs-pial", hemi="lh")

    assert path == fs_root / "sub-02" / "surf" / "lh.pial"
    np.testing.assert_array_equal(loaded_coords, coords)
    np.testing.assert_array_equal(loaded_faces, faces)


def test_save_morph_round_trip(dm):
    values = np.array([9.0, 8.0, 7.0], dtype=np.float32)

    dm.save(values, "01", "fs-thickness", hemi="rh", overwrite=True)

    np.testing.assert_array_equal(dm.load("01", "fs-thickness", hemi="rh"), values)


@pytest.mark.parametrize("key, inputs", [("fs-cortex", {"hemi": "lh"}), ("fs-ras2tkras", {})])
def test_read_only_datatypes_refuse_save(dm, key, inputs):
    with pytest.raises(TypeError, match="read-only"):
        dm.save(object(), "01", key, overwrite=True, **inputs)


# --- swapping readers ------------------------------------------------------


def test_register_reader_override(bids_root, fs_root):
    manager = DatasetManager(bids_root)

    freesurfer.register(manager, root_path=fs_root, readers={"t1": lambda p: ("custom", p.name)})

    assert manager.load("01", "fs-t1") == ("custom", "T1.mgz")
    assert manager.load("01", "fs-aseg").shape == (5, 5, 5)


def test_ras2tkras_save_leaves_t1_untouched(dm, fs_root):
    t1 = fs_root / "sub-01" / "mri" / "T1.mgz"
    before = t1.read_bytes()

    with pytest.raises(TypeError):
        dm.save(np.eye(4), "01", "fs-ras2tkras", overwrite=True)

    assert t1.read_bytes() == before


def test_describe_shows_bound_namer(dm):
    assert dm.describe("fs-pial")["namer"] == "build_fs_path(datatype='pial')"
