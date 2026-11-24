"""Shared pytest fixtures: synthetic datasets built under tmp_path."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def bids_root(tmp_path):
    """Minimal 2-subject BIDS dataset: T1w niftis + dataset_description.json."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Test dataset", "BIDSVersion": "1.8.0"})
    )

    for subject in ("01", "02"):
        anat = root / f"sub-{subject}" / "anat"
        anat.mkdir(parents=True)
        img = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
        nib.save(img, anat / f"sub-{subject}_T1w.nii.gz")

    return root


def _write_label(path: Path, vertices) -> None:
    """Write a minimal FreeSurfer ascii label file.

    nibabel can read these (`nib.freesurfer.io.read_label`) but has no
    writer, so fixtures build the format directly.
    """
    lines = ["#!ascii label", str(len(vertices))]
    lines += [f"{v} 0.0 0.0 0.0 0.0" for v in vertices]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def fs_root(tmp_path):
    """Minimal single-subject FreeSurfer recon-all layout."""
    root = tmp_path / "fs"
    sub = root / "sub-01"
    (sub / "mri").mkdir(parents=True)
    (sub / "surf").mkdir(parents=True)
    (sub / "label").mkdir(parents=True)

    img = nib.MGHImage(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
    nib.save(img, sub / "mri" / "T1.mgz")
    aseg = nib.MGHImage(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4))
    nib.save(aseg, sub / "mri" / "aseg.mgz")

    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    for hemi in ("lh", "rh"):
        nib.freesurfer.io.write_geometry(sub / "surf" / f"{hemi}.white", coords, faces)
        nib.freesurfer.io.write_geometry(sub / "surf" / f"{hemi}.pial", coords, faces)
        nib.freesurfer.io.write_geometry(sub / "surf" / f"{hemi}.sphere", coords, faces)
        nib.freesurfer.io.write_morph_data(
            sub / "surf" / f"{hemi}.area", np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        nib.freesurfer.io.write_morph_data(
            sub / "surf" / f"{hemi}.curv", np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        nib.freesurfer.io.write_morph_data(
            sub / "surf" / f"{hemi}.thickness", np.array([2.0, 2.5, 3.0], dtype=np.float32)
        )
        nib.freesurfer.io.write_morph_data(
            sub / "surf" / f"{hemi}.sulc", np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        )
        _write_label(sub / "label" / f"{hemi}.cortex.label", [0, 1])

    return root


@pytest.fixture
def charm_root(tmp_path):
    """Minimal single-subject SimNIBS charm layout."""
    root = tmp_path / "charm"
    m2m = root / "sub-01" / "m2m_sub-01"
    (m2m / "segmentation").mkdir(parents=True)

    seg = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4))
    nib.save(seg, m2m / "segmentation" / "labeling.nii.gz")

    t1 = nib.Nifti1Image(np.full((3, 3, 3), 2.0, dtype=np.float32), np.eye(4))
    nib.save(t1, m2m / "T1.nii.gz")

    return root


@pytest.fixture
def raw_root(bids_root):
    """`bids_root` plus raw MEG/EEG and a made-up modality (empty placeholder files).

    sub-01/ses-a/meg/sub-01_ses-a_task-x_run-01_meg.fif   one FIF run, zero-padded
    sub-01/ses-a/meg/sub-01_ses-a_task-y_meg.ds/           a CTF folder recording
    sub-01/eeg/sub-01_task-x_eeg.vhdr/.eeg/.vmrk/.json     BrainVision + companions
    sub-01/foo/sub-01_task-x_acq-y_bar.xyz                 a modality BIDS doesn't define
    """
    sub = bids_root / "sub-01"
    meg = sub / "ses-a" / "meg"
    meg.mkdir(parents=True)
    (meg / "sub-01_ses-a_task-x_run-01_meg.fif").touch()
    ds = meg / "sub-01_ses-a_task-y_meg.ds"
    ds.mkdir()
    (ds / "sub-01_ses-a_task-y_meg.meg4").touch()

    eeg = sub / "eeg"
    eeg.mkdir()
    for ext in (".vhdr", ".eeg", ".vmrk"):
        (eeg / f"sub-01_task-x_eeg{ext}").touch()
    (eeg / "sub-01_task-x_eeg.json").write_text("{}")  # pybids parses sidecars

    (sub / "foo").mkdir()
    (sub / "foo" / "sub-01_task-x_acq-y_bar.xyz").touch()
    return bids_root
