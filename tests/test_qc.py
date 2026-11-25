"""Unit tests on small synthetic volumes (no real or demo data needed)."""
import json

import nibabel as nib
import numpy as np
import pytest

from t1qc import metrics, pipeline, rules

CFG = pipeline.load_config()
RNG = np.random.default_rng(0)


def phantom(shape=(100, 120, 100), zooms=(1.5, 1.5, 1.5), noise=5.0):
    """An ellipsoid 'head' on a noisy background, centred in the volume."""
    grid = np.indices(shape).astype(float)
    centre = [(s - 1) / 2 for s in shape]
    r = sum(((g - c) / (0.35 * s)) ** 2 for g, c, s in zip(grid, centre, shape))
    data = np.where(r < 1, 1000.0, 0.0) + np.abs(RNG.normal(0, noise, shape))
    affine = np.diag(list(zooms) + [1.0])
    return nib.Nifti1Image(data.astype("float32"), affine)


def check(img, meta=None):
    meta = meta or {"SeriesDescription": "SAG MPRAGE"}
    data = img.get_fdata()
    geom = metrics.geometry_metrics(img)
    inten = metrics.intensity_metrics(data)
    return rules.evaluate(meta, geom, inten, CFG)


def test_clean_phantom_passes():
    assert check(phantom()).status == "pass"


def test_otsu_separates_two_classes():
    vals = np.concatenate([np.full(500, 10.0), np.full(500, 100.0)])
    assert 10 < metrics.otsu_threshold(vals) < 100


@pytest.mark.parametrize("desc,expected", [
    ("SAG MPRAGE", "t1"), ("3D SPGR", "t1"), ("AX T2 FLAIR", "other"),
    ("T1 FLAIR", "other"), ("localizer", "other"), ("", "unknown"),
])
def test_sequence_classification(desc, expected):
    assert rules.classify_sequence({"SeriesDescription": desc}, CFG["sequence"]) == expected


def test_thick_slices_fail():
    r = check(phantom(shape=(100, 120, 40), zooms=(1.5, 1.5, 5.0)))
    assert r.status == "fail"
    assert any("thick slices" in x for x in r.reasons)


def test_nan_voxels_fail():
    img = phantom()
    data = img.get_fdata()
    data[10:20, 10:20, 10:20] = np.nan
    r = check(nib.Nifti1Image(data, img.affine))
    assert any("corrupted" in x for x in r.reasons)


def test_low_snr_flagged():
    assert check(phantom(noise=300.0)).status in ("warn", "fail")


def test_cut_off_head_fails():
    img = phantom()
    data = img.get_fdata()[:, :, :55]  # chop through the middle of the ellipsoid
    r = check(nib.Nifti1Image(data, img.affine))
    assert any("cut off" in x for x in r.reasons)


def test_contrast_warns():
    r = check(phantom(), {"SeriesDescription": "MPRAGE POST", "ContrastBolusAgent": "Gadobutrol"})
    assert r.status == "warn"


def test_pipeline_end_to_end(tmp_path):
    src = tmp_path / "scans"
    src.mkdir()
    nib.save(phantom(), src / "sub-01_T1w.nii.gz")
    (src / "sub-01_T1w.json").write_text(json.dumps({"SeriesDescription": "MPRAGE", "Manufacturer": "Siemens"}))
    (src / "sub-02_T1w.nii.gz").write_bytes(b"not a nifti file")

    df = pipeline.run(src, tmp_path / "out", montages=True)
    status = dict(zip(df["scan_id"], df["qc_status"]))
    assert status == {"sub-01_T1w": "pass", "sub-02_T1w": "fail"}
    for f in ("catalog.csv", "cohort_summary.md", "report.html", "montages/sub-01_T1w.png"):
        assert (tmp_path / "out" / f).exists()
