"""Regression test: pickle.dumps(DatasetManager instance) must raise.

Turns the documented non-picklability constraint (DatasetManager wraps
bids.BIDSLayout, which cannot be pickled) into a regression test instead
of a landmine. See PLAN.md "Tests".
"""

import pickle

import pytest

from fascicle import DatasetManager


def test_pickle_dumps_raises(bids_root):
    dm = DatasetManager(bids_root)

    with pytest.raises(Exception):
        pickle.dumps(dm)
