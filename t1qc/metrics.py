"""Geometry and image-quality metrics computed from the voxel data.

These are deliberately simple and dependency-light: they are meant to catch
scans that would break or silently degrade a downstream segmentation or
classification model, not to replace a full tool such as MRIQC.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the threshold that best separates two intensity classes."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[np.argmax(between)])


def geometry_metrics(img: nib.Nifti1Image) -> dict:
    shape = img.shape
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    extent = [s * z for s, z in zip(shape[:3], zooms)]
    return {
        "ndim": len(shape),
        "shape": "x".join(str(s) for s in shape),
        "voxel_mm": "x".join(f"{z:.2f}" for z in zooms),
        "max_voxel_mm": max(zooms),
        "anisotropy": max(zooms) / min(zooms),
        "min_fov_mm": min(extent),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
    }


def intensity_metrics(data: np.ndarray) -> dict:
    """Foreground/background split and a basic SNR estimate."""
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    nonfinite_fraction = 1.0 - finite.mean()
    vals = data[finite]

    out = {"nonfinite_fraction": float(nonfinite_fraction)}
    if vals.size == 0 or np.ptp(vals) == 0:
        out.update(foreground_fraction=0.0, snr=0.0, edge_foreground=0.0)
        return out

    thr = otsu_threshold(vals)
    fg = finite & (data > thr)
    bg = finite & ~fg

    fg_mean = float(data[fg].mean()) if fg.any() else 0.0
    bg_std = float(data[bg].std()) if bg.sum() > 10 else 0.0
    # Skull-stripped or zero-padded volumes can have a perfectly flat
    # background; floor the noise estimate so SNR stays finite.
    snr = fg_mean / max(bg_std, 1e-3 * max(fg_mean, 1e-6))

    out.update(
        foreground_fraction=float(fg.mean()),
        snr=float(snr),
        edge_foreground=edge_foreground_fraction(fg),
        p01=float(np.percentile(vals, 1)),
        p99=float(np.percentile(vals, 99)),
    )
    return out


def edge_foreground_fraction(mask: np.ndarray) -> float:
    """Largest share of any outer face of the volume that is covered by foreground.

    A well-framed head scan has background on all six faces. If the field of
    view clipped the head (top of the brain, cerebellum), the face where the
    cut happened shows a solid cross-section of tissue.
    """
    if mask.ndim < 3 or not mask.any():
        return 0.0
    faces = [mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1]]
    return float(max(f.mean() for f in faces))
