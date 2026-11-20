"""Tests for fascicle.extensions.charm.

Covers extension registration, path building, and asserts the seg loader
returns a plain nibabel image, not any wrapper class (locks in the
"no Parcellation" design decision). See PLAN.md "Tests".
"""

import nibabel as nib
import numpy as np
import pytest

from fascicle import DatasetManager
from fascicle.extensions import charm


# --- build_charm_path (operates on bare, unprefixed datatype names) -------


@pytest.mark.parametrize(
    "datatype, expected_suffix",
    [
        ("seg", "sub-01/m2m_sub-01/segmentation/labeling.nii.gz"),
        ("lut", "sub-01/m2m_sub-01/segmentation/labeling_LUT.txt"),
        ("t1", "sub-01/m2m_sub-01/T1.nii.gz"),
    ],
)
def test_build_charm_path_datatype(datatype, expected_suffix):
    path = charm.build_charm_path("/charm", "01", datatype)
    assert str(path) == f"/charm/{expected_suffix}"


def test_build_charm_path_uses_stripped_id_in_both_path_components():
    """Regression test for the mindmap inconsistency: the original built
    sub-{sub_id} (stripped) but m2m_sub-{subject} (unstripped), so
    passing "sub-01" produced a broken double-prefixed path."""
    with_prefix = charm.build_charm_path("/charm", "sub-01", "seg")
    without_prefix = charm.build_charm_path("/charm", "01", "seg")
    assert with_prefix == without_prefix
    assert "m2m_sub-sub-01" not in str(with_prefix)


def test_build_charm_path_unknown_datatype_raises():
    with pytest.raises(ValueError):
        charm.build_charm_path("/charm", "01", "bogus")


# --- register (registry keys are prefixed f"{root_key}-<name>") ----------


def test_register_missing_root_raises(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError):
        charm.register(dm)


def test_register_with_root_path_adds_root(bids_root, charm_root):
    dm = DatasetManager(bids_root)

    charm.register(dm, root_key="charm", root_path=charm_root)

    assert dm.roots["charm"] == charm_root


def test_register_with_existing_root_key(bids_root, charm_root):
    dm = DatasetManager(bids_root)
    dm.add_root("charm", charm_root)

    charm.register(dm, root_key="charm")

    assert "charm-seg" in dm.registry


def test_register_keys_use_custom_root_key_as_prefix(bids_root, charm_root):
    """Registering twice under different root_keys must not collide."""
    dm = DatasetManager(bids_root)

    charm.register(dm, root_key="charm", root_path=charm_root)
    charm.register(dm, root_key="charm2", root_path=charm_root)

    assert "charm-seg" in dm.registry
    assert "charm2-seg" in dm.registry


def test_register_namespace_decoupled_from_root_key(bids_root, charm_root):
    """namespace= lets the key prefix differ from the root's own name."""
    dm = DatasetManager(bids_root)

    charm.register(dm, root_key="my_custom_root", root_path=charm_root, namespace="charm")

    assert "charm-seg" in dm.registry
    assert "my_custom_root-seg" not in dm.registry


# --- loading -----------------------------------------------------------


@pytest.fixture
def dm(bids_root, charm_root):
    manager = DatasetManager(bids_root)
    charm.register(manager, root_key="charm", root_path=charm_root)
    return manager


def test_load_seg_returns_plain_nibabel_image_not_a_wrapper(dm):
    seg = dm.load("01", "charm-seg")

    assert type(seg) is nib.Nifti1Image
    assert seg.shape == (3, 3, 3)


def test_load_t1(dm):
    t1 = dm.load("01", "charm-t1")

    assert type(t1) is nib.Nifti1Image
    assert t1.shape == (3, 3, 3)


def test_save_seg_round_trip_into_new_subject(dm, charm_root):
    data = np.arange(27, dtype=np.int16).reshape(3, 3, 3)

    path = dm.save(nib.Nifti1Image(data, np.eye(4)), "02", "charm-seg")

    assert path == charm_root / "sub-02" / "m2m_sub-02" / "segmentation" / "labeling.nii.gz"
    np.testing.assert_array_equal(nib.load(path).get_fdata(), data)


def test_register_reader_override(bids_root, charm_root):
    manager = DatasetManager(bids_root)

    charm.register(manager, root_path=charm_root, readers={"seg": lambda p: "custom"})

    assert manager.load("01", "charm-seg") == "custom"
    assert type(manager.load("01", "charm-t1")) is nib.Nifti1Image
