"""Generic YAML config loading/saving.

Holds read_config/write_config, adapted from mindmap's
`mindmap/core/io.py` with mindmap-specific references stripped out.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def read_config(file: str | Path, env: bool = True) -> dict[str, Any]:
    """Read a YAML configuration file.

    Parameters
    ----------
    file: str | Path
        Path to the configuration file.
    env: bool
        If True, also sets each top-level key as an environment variable.

    Returns
    -------
    config: dict[str, Any]
        The parsed configuration, with any string value containing "/"
        converted to a Path.
    """
    with open(file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if env:
            os.environ[key] = str(value)
        if isinstance(value, str) and "/" in value:
            config[key] = Path(value)

    return config


def write_config(fname: str | Path, config: dict[str, Any]) -> None:
    """Write a configuration dict to a YAML file.

    Any Path values are converted to strings for serialization; the
    original `config` dict passed in is left untouched.
    """
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in config.items()
    }

    with open(fname, "w", encoding="utf-8") as outfile:
        yaml.dump(config, outfile, sort_keys=False, default_flow_style=False)
