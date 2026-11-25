"""Build a small demo dataset that imitates the problems found in clinical MRI.

Every scan is derived from the MNI152 T1 template that ships with nilearn,
so no download and no patient data are involved. Each variant simulates one
failure mode that real hospital archives contain, and the JSON sidecars carry
fictional scanner and indication metadata in the format dcm2niix produces.

    python scripts/make_demo_data.py --out data/demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.datasets import load_mni152_template

RNG = np.random.default_rng(42)


def rician(x: np.ndarray, sigma: float) -> np.ndarray:
    """Magnitude MRI noise: Gaussian noise on real and imaginary channels."""
    return np.sqrt((x + RNG.normal(0, sigma, x.shape)) ** 2 + RNG.normal(0, sigma, x.shape) ** 2)


def with_affine_zooms(affine: np.ndarray, zooms) -> np.ndarray:
    a = affine.copy()
    for ax, z in enumerate(zooms):
        a[:3, ax] = a[:3, ax] / np.linalg.norm(a[:3, ax]) * z
    return a


def thick_slices(x: np.ndarray, factor: int) -> np.ndarray:
    """Average blocks of `factor` slices along z, like a 2D clinical acquisition."""
    nz = x.shape[2] // factor * factor
    return x[:, :, :nz].reshape(x.shape[0], x.shape[1], -1, factor).mean(axis=3)


def ghosting(x: np.ndarray, strength: float, shift: int) -> np.ndarray:
    """Motion ghosts: faint shifted copies along the phase-encode axis."""
    return x + strength * (np.roll(x, shift, axis=1) + np.roll(x, -shift, axis=1))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tpl = load_mni152_template(resolution=1)
    base = tpl.get_fdata(dtype="float32")
    base = base / np.percentile(base[base > 0], 99) * 1000  # arbitrary scanner units
    aff = tpl.affine
    s = 20.0  # baseline noise level

    def save(name, data, affine, meta):
        nib.save(nib.Nifti1Image(data.astype("float32"), affine), out / f"{name}.nii.gz")
        if meta is not None:
            (out / f"{name}.json").write_text(json.dumps(meta, indent=2))

    def meta(desc, mfr, field, acq, thick, indication, age, sex, contrast=None):
        m = {"SeriesDescription": desc, "Manufacturer": mfr, "MagneticFieldStrength": field,
             "MRAcquisitionType": acq, "SliceThickness": thick, "ScanIndication": indication,
             "PatientAge": age, "PatientSex": sex}
        if contrast:
            m["ContrastBolusAgent"] = contrast
        return m

    # 1. Research-grade 3D T1: should pass.
    save("sub-001_T1w", rician(base, s), aff,
         meta("SAG MPRAGE 1mm", "Siemens", 3.0, "3D", 1.0, "headache", "65-69", "F"))
    # 2. Another good scan, different vendor and field strength.
    save("sub-002_T1w", rician(base, s * 1.5), aff,
         meta("3D SPGR", "GE", 1.5, "3D", 1.0, "dizziness", "70-74", "M"))
    # 3. Typical clinical 2D axial T1 with 5 mm slices: fail for volumetry.
    save("sub-003_T1w", rician(thick_slices(base, 5), s), with_affine_zooms(aff, (1, 1, 5)),
         meta("AX T1 SE", "Philips", 1.5, "2D", 5.0, "suspected minor stroke", "75-79", "F"))
    # 4. 2 mm slices: usable by SynthSeg, not ideal for FreeSurfer: warn.
    save("sub-004_T1w", rician(thick_slices(base, 2), s), with_affine_zooms(aff, (1, 1, 2)),
         meta("T1 TFE", "Philips", 3.0, "3D", 2.0, "headache", "60-64", "M"))
    # 5. Motion: ghosting plus extra noise lowers SNR.
    save("sub-005_T1w", rician(ghosting(base, 0.3, 20), s * 12), aff,
         meta("SAG MPRAGE", "Siemens", 3.0, "3D", 1.0, "dizziness", "80-84", "F"))
    # 6. Head cut off: field of view clipped at the top of the brain.
    save("sub-006_T1w", rician(base[:, :, : base.shape[2] - 55], s), aff,
         meta("SAG MPRAGE", "Siemens", 1.5, "3D", 1.0, "headache", "65-69", "M"))
    # 7. Corrupted: a block of NaN voxels from a bad reconstruction or export.
    bad = rician(base, s)
    bad[60:90, 80:120, 60:90] = np.nan
    save("sub-007_T1w", bad, aff, meta("3D SPGR", "GE", 3.0, "3D", 1.0, "suspected minor stroke", "70-74", "F"))
    # 8. Wrong sequence: a FLAIR filed in the same session.
    # Crude FLAIR-like contrast: tissue intensities reordered, background still dark.
    flair = np.where(base > 0, 0.9 * base.max() - 0.5 * base, 0)
    save("sub-008_FLAIR", rician(flair, s), aff,
         meta("AX T2 FLAIR", "Siemens", 3.0, "2D", 1.0, "suspected minor stroke", "75-79", "M"))
    # 9. Post-contrast T1: usable image, but differs from the model's training data.
    save("sub-009_T1w", rician(base, s), aff,
         meta("SAG MPRAGE POST", "Siemens", 3.0, "3D", 1.0, "headache", "60-64", "F", contrast="Gadobutrol"))
    # 10. No sidecar at all: sequence unknown.
    save("sub-010_T1w", rician(base, s), aff, None)
    # 11. Truncated file: a transfer that stopped halfway.
    raw = (out / "sub-001_T1w.nii.gz").read_bytes()
    (out / "sub-011_T1w.nii.gz").write_bytes(raw[: len(raw) // 3])
    (out / "sub-011_T1w.json").write_text(json.dumps(
        meta("SAG MPRAGE", "GE", 1.5, "3D", 1.0, "dizziness", "70-74", "M"), indent=2))

    print(f"wrote 11 demo scans to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo", type=Path)
    main(ap.parse_args().out)
