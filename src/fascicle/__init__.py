"""Fascicle: a BIDS-native dataset manager with pluggable extensions for derivatives."""

from .datatype import Datatype
from .extension import register_extension
from .manager import DatasetManager
from .naming import bids_pattern, derivative_namer

__all__ = ["DatasetManager", "Datatype", "bids_pattern", "derivative_namer", "register_extension"]
