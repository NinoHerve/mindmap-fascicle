"""Tests for fascicle.config: read_config env/Path handling behavior."""

import os
from pathlib import Path

from fascicle.config import read_config, write_config


# --- read_config ---------------------------------------------------------


def test_read_config_converts_slash_strings_to_path(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("DIR_RAW: /data/bids\nNAME: demo\n")

    config = read_config(cfg_path, env=False)

    assert config["DIR_RAW"] == Path("/data/bids")
    assert isinstance(config["DIR_RAW"], Path)
    assert config["NAME"] == "demo"
    assert isinstance(config["NAME"], str)


def test_read_config_non_string_values_untouched(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("COUNT: 3\nFLAG: true\nITEMS:\n  - a\n  - b\n")

    config = read_config(cfg_path, env=False)

    assert config["COUNT"] == 3
    assert config["FLAG"] is True
    assert config["ITEMS"] == ["a", "b"]


def test_read_config_env_true_sets_environ(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("FASCICLE_TEST_VAR: hello\n")
    os.environ.pop("FASCICLE_TEST_VAR", None)

    try:
        read_config(cfg_path, env=True)
        assert os.environ["FASCICLE_TEST_VAR"] == "hello"
    finally:
        os.environ.pop("FASCICLE_TEST_VAR", None)


def test_read_config_env_true_stringifies_non_str_values(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("FASCICLE_TEST_COUNT: 3\n")
    os.environ.pop("FASCICLE_TEST_COUNT", None)

    try:
        read_config(cfg_path, env=True)
        assert os.environ["FASCICLE_TEST_COUNT"] == "3"
    finally:
        os.environ.pop("FASCICLE_TEST_COUNT", None)


def test_read_config_env_false_does_not_set_environ(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("FASCICLE_TEST_VAR2: hello\n")
    os.environ.pop("FASCICLE_TEST_VAR2", None)

    read_config(cfg_path, env=False)

    assert "FASCICLE_TEST_VAR2" not in os.environ


# --- write_config --------------------------------------------------------


def test_write_config_round_trip(tmp_path):
    cfg_path = tmp_path / "config.yaml"

    write_config(cfg_path, {"DIR_RAW": Path("/data/bids"), "NAME": "demo"})
    loaded = read_config(cfg_path, env=False)

    assert loaded["DIR_RAW"] == Path("/data/bids")
    assert loaded["NAME"] == "demo"


def test_write_config_does_not_mutate_input(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    original = {"DIR_RAW": Path("/data/bids")}

    write_config(cfg_path, original)

    assert original["DIR_RAW"] == Path("/data/bids")
    assert isinstance(original["DIR_RAW"], Path)


def test_write_config_preserves_key_order(tmp_path):
    cfg_path = tmp_path / "config.yaml"

    write_config(cfg_path, {"Z_KEY": "1", "A_KEY": "2"})
    text = cfg_path.read_text()

    assert text.index("Z_KEY") < text.index("A_KEY")
