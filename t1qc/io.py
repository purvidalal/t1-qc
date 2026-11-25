"""Discover scans and read their metadata.

Supported inputs:
  * NIfTI files (.nii / .nii.gz), optionally with a BIDS-style JSON sidecar
    (the format dcm2niix writes), e.g. sub-001_T1w.nii.gz + sub-001_T1w.json
  * DICOM headers via `read_dicom_series_header`, for cataloging a series
    before conversion.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib

# Sidecar / DICOM fields worth carrying into the catalog.
METADATA_FIELDS = [
    "Manufacturer",
    "ManufacturersModelName",
    "MagneticFieldStrength",
    "SeriesDescription",
    "ProtocolName",
    "MRAcquisitionType",   # "2D" or "3D"
    "SliceThickness",
    "ContrastBolusAgent",
    "PatientAge",          # de-identified / binned upstream
    "PatientSex",
    "ScanIndication",      # not a DICOM tag; joined from the order record upstream
]


def find_scans(root: str | Path) -> list[Path]:
    """Return every NIfTI file under `root`, sorted for reproducibility."""
    root = Path(root)
    found = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    return sorted(found)


def scan_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def read_sidecar(path: Path) -> dict:
    """Read the JSON sidecar next to a NIfTI file, if there is one."""
    sidecar = path.parent / f"{scan_id(path)}.json"
    if not sidecar.exists():
        return {}
    with open(sidecar) as f:
        data = json.load(f)
    return {k: data.get(k) for k in METADATA_FIELDS if k in data}


def load_nifti(path: Path):
    """Load a NIfTI image. Returns (image, error_message)."""
    try:
        img = nib.load(str(path))
        img.get_fdata(dtype="float32")  # force a read so truncated files fail here
        return img, None
    except Exception as exc:  # corrupted / truncated / unreadable
        return None, f"unreadable: {type(exc).__name__}: {exc}"


def read_dicom_series_header(series_dir: str | Path) -> dict:
    """Read catalog fields from the first DICOM file in a series folder.

    Pixel data is skipped (stop_before_pixels), so this is fast enough to
    catalog tens of thousands of series before deciding what to convert.
    """
    import pydicom

    files = sorted(p for p in Path(series_dir).iterdir() if p.is_file())
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        out = {k: ds.get(k) for k in METADATA_FIELDS if k in ds}
        out["NumFiles"] = len(files)
        # Convert pydicom value types to plain Python for CSV output.
        return {k: (str(v) if v is not None else None) for k, v in out.items()}
    return {"NumFiles": len(files), "error": "no readable DICOM files"}
