"""Outputs: per-scan montages, the catalog CSV, cohort summaries, an HTML report."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATUS_COLOR = {"pass": "#2e7d32", "warn": "#ef6c00", "fail": "#c62828"}


def save_montage(data: np.ndarray, zooms, out_path: Path, title: str) -> None:
    """Save mid-axial, mid-coronal and mid-sagittal slices side by side.

    Voxel sizes are used for the aspect ratio, so thick-slice scans look as
    stretched as they really are, which is the point of a visual check.
    """
    data = np.nan_to_num(np.asarray(data, dtype=np.float32))
    if data.ndim > 3:
        data = data[..., 0]
    lo, hi = np.percentile(data, [1, 99.5])
    zx, zy, zz = (list(zooms) + [1, 1, 1])[:3]
    cx, cy, cz = (s // 2 for s in data.shape)

    views = [
        ("sagittal", data[cx, :, :].T, zz / zy),
        ("coronal", data[:, cy, :].T, zz / zx),
        ("axial", data[:, :, cz].T, zy / zx),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="black")
    for ax, (name, sl, aspect) in zip(axes, views):
        ax.imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect=aspect)
        ax.set_title(name, color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, color="white", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80, facecolor="black")
    plt.close(fig)


def cohort_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Counts of QC status by scanner, field strength and scan indication."""
    tables = {}
    for col, label in [
        ("Manufacturer", "By scanner manufacturer"),
        ("MagneticFieldStrength", "By field strength (T)"),
        ("MRAcquisitionType", "By acquisition (2D/3D)"),
        ("ScanIndication", "By scan indication"),
        ("PatientSex", "By sex"),
    ]:
        if col in df and df[col].notna().any():
            t = pd.crosstab(df[col].fillna("unknown"), df["qc_status"], margins=True, margins_name="total")
            tables[label] = t.reindex(columns=[c for c in ["pass", "warn", "fail", "total"] if c in t.columns])
    return tables


def write_html(df: pd.DataFrame, tables: dict, out_dir: Path) -> Path:
    counts = df["qc_status"].value_counts().reindex(["pass", "warn", "fail"], fill_value=0)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>T1 QC report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px;color:#222}"
        "table{border-collapse:collapse;margin:0.5rem 0 1.5rem}td,th{border:1px solid #ddd;padding:4px 8px;font-size:13px}"
        "th{background:#f5f5f5}.card{border:1px solid #ddd;border-radius:6px;padding:0.6rem;margin:0.8rem 0}"
        ".badge{color:white;border-radius:4px;padding:1px 6px;font-size:12px}img{max-width:100%}</style></head><body>",
        "<h1>T1 MRI quality-control report</h1>",
        f"<p><b>{len(df)}</b> scans: "
        + " · ".join(f"<span class='badge' style='background:{STATUS_COLOR[s]}'>{s} {n}</span>" for s, n in counts.items())
        + "</p>",
    ]
    for label, t in tables.items():
        parts += [f"<h3>{html.escape(label)}</h3>", t.to_html()]
    parts.append("<h2>Scans</h2>")
    for _, row in df.sort_values(["qc_status", "scan_id"], key=lambda s: s.map({"fail": 0, "warn": 1, "pass": 2}) if s.name == "qc_status" else s).iterrows():
        reasons = "<br>".join(html.escape(r) for r in str(row["qc_reasons"]).split(" | ") if r) or "all checks passed"
        img = f"<img src='montages/{html.escape(row['scan_id'])}.png'>" if row.get("montage") else ""
        parts.append(
            f"<div class='card'><b>{html.escape(row['scan_id'])}</b> "
            f"<span class='badge' style='background:{STATUS_COLOR[row['qc_status']]}'>{row['qc_status']}</span>"
            f" <small>{html.escape(str(row.get('voxel_mm', '')))} mm · {html.escape(str(row.get('Manufacturer', '')))}"
            f" {html.escape(str(row.get('MagneticFieldStrength', '')))}T · SNR {row.get('snr', float('nan')):.1f}</small>"
            f"<br><small>{reasons}</small>{img}</div>"
        )
    parts.append("</body></html>")
    out = out_dir / "report.html"
    out.write_text("\n".join(parts))
    return out
