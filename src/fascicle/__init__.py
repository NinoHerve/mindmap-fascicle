"""Fascicle: a BIDS-native dataset manager with pluggable extensions for derivatives."""

from .datatype import Datatype, RawDatatype
from .extension import register_extension, register_raw
from .manager import DatasetManager
from .naming import bids_pattern, derivative_namer

__all__ = [
    "DatasetManager",
    "Datatype",
    "RawDatatype",
    "bids_pattern",
    "derivative_namer",
    "register_extension",
    "register_raw",
]
