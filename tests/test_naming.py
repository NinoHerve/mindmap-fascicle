"""Tests for fascicle.naming: bids_pattern / derivative_namer."""

import json

import nibabel as nib
import numpy as np
import pytest

from fascicle import DatasetManager, Datatype, derivative_namer, register_extension
from fascicle.naming import BIDS_ENTITIES, bids_pattern


# --- the entity table matches the BIDS spec ----------------------------------


def test_entity_table_matches_installed_bids_schema():
    """Order, filename keys, formats and allowed values all come from the spec."""
    schema = pytest.importorskip("bidsschematools.schema").load_schema()
    spec = [e for e in schema.rules.entities if e != "subject"]

    assert [key for _, key, _, _ in BIDS_ENTITIES] == [
        schema.objects.entities[e].name for e in spec
    ]
    for (_, _, fmt, allowed), e in zip(BIDS_ENTITIES, spec):
        assert fmt == schema.objects.entities[e]["format"]
        assert allowed == (tuple(schema.objects.entities[e]["enum"]) if "enum" in schema.objects.entities[e] else None)


def test_keyword_names_match_pybids_where_pybids_has_the_entity():
    from importlib.resources import files

    config = files("bids.layout") / "config"
    pybids = {}
    for name in ("bids.json", "derivatives.json"):
        for ent in json.loads((config / name).read_text())["entities"]:
            pybids[ent["name"]] = ent["pattern"]

    for name, key, _, _ in BIDS_ENTITIES:
        if name in pybids:
            assert f"{key}-" in pybids[name], name


# --- pattern generation ------------------------------------------------------


def test_pattern_shape():
    pattern = bids_pattern("mask", "anat", (".nii.gz", ".json"), entities=("session", "desc"))

    assert pattern == (
        "sub-{subject}[/ses-{session}]/anat/"
        "sub-{subject}[_ses-{session}][_desc-{desc}]_mask{extension<.nii.gz|.json>|.nii.gz}"
    )


def test_pattern_without_session_datatype_or_extension():
    pattern = bids_pattern("stc", None, None, entities=("task",))

    assert pattern == "sub-{subject}/sub-{subject}[_task-{task}]_stc"


@pytest.mark.parametrize("bad", ["", "my_mask", "a/b", "x{y}", "has space"])
def test_bad_suffix_or_datatype_raises(bad):
    with pytest.raises(ValueError):
        bids_pattern(bad)
    with pytest.raises(ValueError):
        bids_pattern("mask", datatype=bad)


def test_unknown_entity_in_declaration_raises():
    with pytest.raises(ValueError, match="bogus"):
        bids_pattern("mask", entities=("desc", "bogus"))


@pytest.mark.parametrize("extra", ["desc", "ses", "subject", "bad-name"])
def test_extra_entity_clashing_or_invalid_raises(extra):
    with pytest.raises(ValueError):
        bids_pattern("mask", extra_entities=(extra,))


# --- naming --------------------------------------------------------------------


@pytest.fixture
def mask_namer():
    return derivative_namer("mask", datatype="anat", extensions=(".nii.gz", ".json"))


def test_entities_written_in_spec_order_whatever_the_kwarg_order(tmp_path, mask_namer):
    path = mask_namer(tmp_path, "01", desc="brain", space="MNI152", run=2, session="a")

    assert path == tmp_path / "sub-01/ses-a/anat/sub-01_ses-a_run-2_space-MNI152_desc-brain_mask.nii.gz"


def test_sidecar_extension_and_sub_prefix(tmp_path, mask_namer):
    path = mask_namer(tmp_path, "sub-01", extension=".json")

    assert path == tmp_path / "sub-01/anat/sub-01_mask.json"


def test_extra_entity_placed_before_desc(tmp_path):
    namer = derivative_namer("mask", "anat", extra_entities=("cut",))

    path = namer(tmp_path, "01", desc="brain", cut="axial", space="T1w")

    assert path.name == "sub-01_space-T1w_cut-axial_desc-brain_mask.nii.gz"


def test_restricted_entities_reject_others(tmp_path):
    namer = derivative_namer("conn", "connectivity", (".npy",), entities=("atlas", "desc"))

    assert namer(tmp_path, "01", atlas="DK").name == "sub-01_atlas-DK_conn.npy"
    with pytest.raises(ValueError, match="session"):
        namer(tmp_path, "01", session="a")


@pytest.mark.parametrize(
    "values, match",
    [
        ({"bogus": "x"}, "bogus"),
        ({"suffix": "T1w"}, "suffix"),
        ({"datatype": "func"}, "datatype"),
        ({"extension": ".npy"}, ".npy"),
        ({"desc": "brain-mask"}, "alphanumeric"),
        ({"desc": "brain_mask"}, "alphanumeric"),
        ({"run": "one"}, "integer"),
        ({"hemi": "lh"}, "L"),
        ({"part": "magnitude"}, "mag"),
    ],
)
def test_invalid_input_raises(tmp_path, mask_namer, values, match):
    with pytest.raises(ValueError, match=match):
        mask_namer(tmp_path, "01", **values)


def test_none_values_are_skipped(tmp_path, mask_namer):
    assert mask_namer(tmp_path, "01", desc=None).name == "sub-01_mask.nii.gz"


def test_hyphenated_suffix(tmp_path):
    namer = derivative_namer("i-ave", "ave", (".fif",))

    assert namer(tmp_path, "01").name == "sub-01_i-ave.fif"


# --- fixed entities ------------------------------------------------------------


def test_fixed_custom_entity_is_written_and_cannot_be_passed(tmp_path):
    namer = derivative_namer("ave", "ave", (".fif",), extra_entities=("signal",), fixed={"signal": "virtual"})

    assert namer(tmp_path, "01", desc="x").name == "sub-01_signal-virtual_desc-x_ave.fif"
    with pytest.raises(ValueError, match="Fixed for this datatype"):
        namer(tmp_path, "01", signal="intracranial")


def test_fixed_spec_entity_kept_in_spec_order_even_with_restricted_entities(tmp_path):
    namer = derivative_namer("mask", "anat", entities=("desc",), fixed={"space": "MNI152"})

    assert namer(tmp_path, "01", desc="brain").name == "sub-01_space-MNI152_desc-brain_mask.nii.gz"


def test_fixed_session_goes_in_folder_too(tmp_path):
    namer = derivative_namer("mask", "anat", fixed={"session": "postop"})

    assert namer(tmp_path, "01") == tmp_path / "sub-01/ses-postop/anat/sub-01_ses-postop_mask.nii.gz"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fixed": {"signal": "virtual"}}, "extra_entities"),  # custom, undeclared
        ({"fixed": {"desc": "brain-mask"}}, "alphanumeric"),
        ({"fixed": {"hemi": "lh"}}, "L"),
    ],
)
def test_invalid_fixed_raises_at_declaration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        derivative_namer("mask", "anat", **kwargs)


# --- through a manager --------------------------------------------------------


def test_round_trip_through_a_toy_extension(bids_root, tmp_path):
    table = {"mask": Datatype(derivative_namer("mask", "anat", (".nii.gz", ".json")))}
    dm = DatasetManager(bids_root)
    register_extension(dm, table, root_key="mytool", root_path=tmp_path / "derivatives" / "mytool")
    data = np.ones((2, 3, 4), dtype=np.float32)

    img_path = dm.save(nib.Nifti1Image(data, np.eye(4)), "01", "mytool-mask", desc="brain")
    json_path = dm.save({"Sources": ["T1w"]}, "01", "mytool-mask", desc="brain", extension=".json")

    assert str(img_path.relative_to(tmp_path)) == "derivatives/mytool/sub-01/anat/sub-01_desc-brain_mask.nii.gz"
    assert json_path.name == "sub-01_desc-brain_mask.json"
    np.testing.assert_array_equal(dm.load("01", "mytool-mask", desc="brain").get_fdata(), data)
    assert dm.load("01", "mytool-mask", desc="brain", extension=".json") == {"Sources": ["T1w"]}


def test_describe_shows_derivative_namer(bids_root):
    dm = DatasetManager(bids_root)
    dm.add_datatype("mask", None, derivative_namer("mask", "anat"))

    assert dm.describe("mask")["namer"] == "derivative_namer('mask')"
