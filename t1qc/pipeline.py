"""Run QC over a folder of scans and build the catalog."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from . import io, metrics, report, rules

log = logging.getLogger("t1qc")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def _markdown_table(t: pd.DataFrame) -> str:
    cols = [str(t.index.name or "")] + [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in t.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def qc_one(path: Path, cfg: dict, montage_dir: Path | None = None) -> dict:
    """QC a single NIfTI file. Never raises: failures become catalog rows."""
    row = {"scan_id": io.scan_id(path), "path": str(path)}
    row.update(io.read_sidecar(path))

    img, err = io.load_nifti(path)
    if img is None:
        row.update(qc_status="fail", qc_reasons=f"fail: {err}", montage=False)
        return row

    data = img.get_fdata(dtype="float32")
    geom = metrics.geometry_metrics(img)
    inten = metrics.intensity_metrics(data)
    row.update(geom)
    row.update(inten)
    row["sequence_class"] = rules.classify_sequence(row, cfg["sequence"])

    result = rules.evaluate(row, geom, inten, cfg)
    row["qc_status"] = result.status
    row["qc_reasons"] = " | ".join(result.reasons)

    row["montage"] = False
    if montage_dir is not None:
        try:
            report.save_montage(
                data, img.header.get_zooms(), montage_dir / f"{row['scan_id']}.png",
                f"{row['scan_id']}  [{result.status}]",
            )
            row["montage"] = True
        except Exception as exc:
            log.warning("montage failed for %s: %s", row["scan_id"], exc)
    return row


def run(input_dir: str | Path, out_dir: str | Path, config: str | Path | None = None,
        montages: bool = True) -> pd.DataFrame:
    cfg = load_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = out_dir / "montages" if montages else None

    paths = io.find_scans(input_dir)
    log.info("found %d scans under %s", len(paths), input_dir)
    rows = []
    for i, p in enumerate(paths, 1):
        row = qc_one(p, cfg, montage_dir)
        log.info("[%d/%d] %s -> %s", i, len(paths), row["scan_id"], row["qc_status"])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("no scans found")
        return df

    lead = ["scan_id", "qc_status", "qc_reasons", "sequence_class"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(out_dir / "catalog.csv", index=False)

    tables = report.cohort_summary(df)
    with open(out_dir / "cohort_summary.md", "w") as f:
        f.write(f"# Cohort summary ({len(df)} scans)\n\n")
        for label, t in tables.items():
            f.write(f"## {label}\n\n{_markdown_table(t)}\n\n")
    report.write_html(df, tables, out_dir)
    log.info("wrote catalog.csv, cohort_summary.md and report.html to %s", out_dir)
    return df
