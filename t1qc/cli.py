"""Command line entry point.

    python -m t1qc.cli data/demo --out results/
    python -m t1qc.cli /path/to/bids --out results/ --config my_thresholds.yaml --no-montages
"""
from __future__ import annotations

import argparse
import logging

from .pipeline import run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="QC and catalog clinical T1-weighted MRI.")
    ap.add_argument("input", help="folder containing .nii/.nii.gz scans (searched recursively)")
    ap.add_argument("--out", default="qc_output", help="output folder")
    ap.add_argument("--config", default=None, help="YAML thresholds (default: config/default.yaml)")
    ap.add_argument("--no-montages", action="store_true", help="skip slice images (faster)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run(args.input, args.out, args.config, montages=not args.no_montages)
    if not df.empty:
        print("\n" + df["qc_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
