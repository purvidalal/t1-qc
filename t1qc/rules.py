"""Turn metadata and metrics into pass / warn / fail decisions with reasons."""
from __future__ import annotations

from dataclasses import dataclass, field

ORDER = {"pass": 0, "warn": 1, "fail": 2}


@dataclass
class QCResult:
    status: str = "pass"
    reasons: list[str] = field(default_factory=list)

    def add(self, status: str, reason: str) -> None:
        self.reasons.append(f"{status}: {reason}")
        if ORDER[status] > ORDER[self.status]:
            self.status = status


def classify_sequence(meta: dict, cfg: dict) -> str:
    """Label a series as 't1', 'other' or 'unknown' from its description."""
    text = " ".join(
        str(meta.get(k) or "") for k in ("SeriesDescription", "ProtocolName")
    ).lower()
    if not text.strip():
        return "unknown"
    # Check exclusions first: "T1 FLAIR" or "T2" should not count as a 3D T1.
    if any(k in text for k in cfg["exclude_keywords"]):
        return "other"
    if any(k in text for k in cfg["t1_keywords"]):
        return "t1"
    return "unknown"


def has_contrast(meta: dict) -> bool:
    agent = str(meta.get("ContrastBolusAgent") or "").strip().lower()
    return agent not in ("", "none", "no", "n/a")


def evaluate(meta: dict, geom: dict, inten: dict, cfg: dict) -> QCResult:
    r = QCResult()
    g, i, c = cfg["geometry"], cfg["intensity"], cfg["coverage"]

    # Sequence: is this the scan the model needs at all?
    seq = classify_sequence(meta, cfg["sequence"])
    if seq == "other":
        r.add("fail", f"not a T1-weighted series ({meta.get('SeriesDescription')})")
    elif seq == "unknown":
        r.add("warn", "sequence type unknown (no recognizable SeriesDescription)")

    # Geometry
    if geom["ndim"] < g["min_dims"] or geom["ndim"] > 3:
        r.add("fail", f"expected a 3D volume, got {geom['ndim']}D")
    if geom["max_voxel_mm"] > g["slice_thickness_fail_mm"]:
        r.add("fail", f"thick slices ({geom['max_voxel_mm']:.1f} mm)")
    elif geom["max_voxel_mm"] > g["slice_thickness_warn_mm"]:
        r.add("warn", f"slices thicker than research grade ({geom['max_voxel_mm']:.1f} mm)")
    if geom["anisotropy"] > g["anisotropy_warn"]:
        r.add("warn", f"anisotropic voxels (ratio {geom['anisotropy']:.1f})")
    if geom["min_fov_mm"] < g["min_fov_mm"]:
        r.add("fail", f"partial coverage (smallest extent {geom['min_fov_mm']:.0f} mm)")

    # Intensity and corruption
    corrupted = inten["nonfinite_fraction"] > i["max_nonfinite_fraction"]
    if corrupted:
        r.add("fail", f"corrupted voxels ({inten['nonfinite_fraction']:.2%} NaN/Inf)")
    if inten["foreground_fraction"] < i["min_foreground_fraction"]:
        r.add("fail", "almost no foreground (empty or blank volume)")
    if not corrupted:  # SNR is meaningless on a corrupted volume
        if inten["snr"] < i["snr_fail"]:
            r.add("fail", f"very low SNR ({inten['snr']:.1f})")
        elif inten["snr"] < i["snr_warn"]:
            r.add("warn", f"low SNR ({inten['snr']:.1f}); noise or motion likely")

    # Field-of-view clipping
    if inten["edge_foreground"] > c["edge_foreground_fail"]:
        r.add("fail", f"head cut off by field of view ({inten['edge_foreground']:.0%} of a border face is tissue)")
    elif inten["edge_foreground"] > c["edge_foreground_warn"]:
        r.add("warn", f"head touches edge of field of view ({inten['edge_foreground']:.0%} of a border face)")

    # Contrast agent
    if has_contrast(meta):
        r.add(cfg["contrast"]["contrast_status"], f"contrast-enhanced ({meta.get('ContrastBolusAgent')})")

    return r
