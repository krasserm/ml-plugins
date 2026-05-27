#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "beautifulsoup4", "whoosh", "python-dotenv"]
# ///
"""HF documentation CLI. Wraps ml-intern's explore/fetch/openapi handlers.

Examples:
  hf_docs.py explore trl --query "SFTTrainer"
  hf_docs.py fetch https://huggingface.co/docs/trl/sft_trainer
  hf_docs.py find-api --query "create repo"
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # silence whoosh's SyntaxWarnings on import

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlhelpers import cli  # noqa: E402

cli.load_env()
from mlhelpers.docs_tools import (  # noqa: E402
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
    search_openapi_handler,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HF documentation research tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explore", help="Search a library's docs")
    e.add_argument("endpoint", help="e.g. trl, transformers, datasets, peft, accelerate")
    e.add_argument("--query", help="Search terms; omit to list the library's doc pages.")
    e.add_argument("--max-results", type=int, help="Max pages to return (default 10).")

    f = sub.add_parser("fetch", help="Fetch full page content")
    f.add_argument("url", help="Doc page URL from an explore result.")

    g = sub.add_parser("find-api", help="Find HF REST API endpoints")
    g.add_argument("--query", help="Search terms for the endpoint.")
    g.add_argument("--tag", help="Filter by API tag, e.g. 'models', 'datasets'.")

    a = p.parse_args()
    session = cli.hf_session()

    if a.cmd == "explore":
        args = {"endpoint": a.endpoint}
        if a.query is not None:
            args["query"] = a.query
        if a.max_results is not None:
            args["max_results"] = a.max_results
        cli.run(explore_hf_docs_handler(args, session=session))
    elif a.cmd == "fetch":
        cli.run(hf_docs_fetch_handler({"url": a.url}, session=session))
    elif a.cmd == "find-api":
        args = {}
        if a.query is not None:
            args["query"] = a.query
        if a.tag is not None:
            args["tag"] = a.tag
        cli.run(search_openapi_handler(args))


if __name__ == "__main__":
    main()
