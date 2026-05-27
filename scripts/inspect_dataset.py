#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""Inspect an HF dataset (schema, splits, sample rows) via the datasets-server.
Wraps ml-intern's hf_inspect_dataset_handler.

Example:
  inspect_dataset.py --dataset krasserm/deepjob-clean-sft-v2 --split train --sample-rows 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlhelpers import cli  # noqa: E402

cli.load_env()
from mlhelpers.dataset_tools import hf_inspect_dataset_handler  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect an HF dataset")
    p.add_argument("--dataset", required=True, help="e.g. org/dataset-name")
    p.add_argument("--config", help="Dataset config/subset name (if the dataset has several).")
    p.add_argument("--split", help="Split to inspect, e.g. train, validation, test.")
    p.add_argument("--sample-rows", type=int, help="Number of example rows to print.")
    a = p.parse_args()

    args = {"dataset": a.dataset}
    if a.config is not None:
        args["config"] = a.config
    if a.split is not None:
        args["split"] = a.split
    if a.sample_rows is not None:
        args["sample_rows"] = a.sample_rows
    cli.run(hf_inspect_dataset_handler(args, session=cli.hf_session()))


if __name__ == "__main__":
    main()
