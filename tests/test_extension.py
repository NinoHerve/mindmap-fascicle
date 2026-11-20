"""Tests for fascicle.register_extension, using a toy extension table."""

from functools import partial

import pytest

from fascicle import DatasetManager, Datatype, register_extension


def _namer(root, subject, datatype):
    return root / f"sub-{subject}_{datatype}.txt"


def _read(path):
    return path.read_text()


def _write(data, path):
    path.write_text(data)


TOY = {
    "a": Datatype(partial(_namer, datatype="a"), _read, _write),
    "b": Datatype(partial(_namer, datatype="b"), _read),  # read-only
}


@pytest.fixture
def toy_root(tmp_path):
    root = tmp_path / "toy"
    root.mkdir()
    (root / "sub-01_a.txt").write_text("A")
    (root / "sub-01_b.txt").write_text("B")
    return root


def test_missing_root_raises_key_error(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(KeyError, match="toy"):
        register_extension(dm, TOY, root_key="toy")


def test_root_path_adds_root_and_default_namespace_is_root_key(bids_root, toy_root):
    dm = DatasetManager(bids_root)

    register_extension(dm, TOY, root_key="toy", root_path=toy_root)

    assert dm.roots["toy"] == toy_root
    assert {"toy-a", "toy-b"} <= set(dm.registry)
    assert dm.registry["toy-a"].root_key == "toy"
    assert dm.load("01", "toy-a") == "A"


def test_existing_root_key_is_reused(bids_root, toy_root):
    dm = DatasetManager(bids_root)
    dm.add_root("toy", toy_root)

    register_extension(dm, TOY, root_key="toy")

    assert dm.load("01", "toy-b") == "B"


def test_explicit_namespace_decoupled_from_root_key(bids_root, toy_root):
    dm = DatasetManager(bids_root)

    register_extension(dm, TOY, root_key="toy", root_path=toy_root, namespace="t")

    assert "t-a" in dm.registry
    assert "toy-a" not in dm.registry


def test_two_registrations_under_different_roots_dont_collide(bids_root, toy_root, tmp_path):
    other = tmp_path / "toy2"
    other.mkdir()
    (other / "sub-01_a.txt").write_text("A2")
    dm = DatasetManager(bids_root)

    register_extension(dm, TOY, root_key="toy", root_path=toy_root)
    register_extension(dm, TOY, root_key="toy2", root_path=other)

    assert dm.load("01", "toy-a") == "A"
    assert dm.load("01", "toy2-a") == "A2"


def test_reader_and_writer_overrides(bids_root, toy_root):
    dm = DatasetManager(bids_root)
    written = {}

    register_extension(
        dm,
        TOY,
        root_key="toy",
        root_path=toy_root,
        readers={"a": lambda p: "overridden"},
        writers={"b": lambda data, p: written.update(data=data)},
    )

    assert dm.load("01", "toy-a") == "overridden"
    assert dm.load("01", "toy-b") == "B"  # untouched reader
    dm.save("new", "01", "toy-b", overwrite=True)
    assert written == {"data": "new"}


def test_overrides_dont_mutate_the_table(bids_root, toy_root):
    dm = DatasetManager(bids_root)

    register_extension(dm, TOY, root_key="toy", root_path=toy_root, readers={"a": lambda p: 0})

    assert TOY["a"].reader is _read
    assert TOY["a"].root_key is None


@pytest.mark.parametrize("arg", ["readers", "writers"])
def test_unknown_override_name_raises_value_error(bids_root, toy_root, arg):
    dm = DatasetManager(bids_root)

    with pytest.raises(ValueError, match="bogus"):
        register_extension(dm, TOY, root_key="toy", root_path=toy_root, **{arg: {"bogus": _read}})


def test_save_through_extension_round_trip(bids_root, toy_root):
    dm = DatasetManager(bids_root)
    register_extension(dm, TOY, root_key="toy", root_path=toy_root)

    path = dm.save("A!", "01", "toy-a", overwrite=True)

    assert path == toy_root / "sub-01_a.txt"
    assert dm.load("01", "toy-a") == "A!"
