#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "beautifulsoup4", "python-dotenv"]
# ///
"""Papers & citations CLI (arXiv + HF + Semantic Scholar). Wraps ml-intern's
hf_papers_handler. Operation-specific args are validated by the handler.

Examples:
  papers.py search --query "GPQA graduate questions" --sort-by citationCount
  papers.py citation-graph --arxiv-id 2311.12022 --direction citations
  papers.py read --arxiv-id 2604.01348 --section 3
  papers.py find-datasets --arxiv-id 2604.01348
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlhelpers import cli  # noqa: E402

cli.load_env()
from mlhelpers.papers_tool import hf_papers_handler  # noqa: E402

# CLI subcommand name -> handler "operation" string
OPS = {
    "search": "search", "trending": "trending", "details": "paper_details",
    "read": "read_paper", "citation-graph": "citation_graph",
    "snippet-search": "snippet_search", "recommend": "recommend",
    "find-datasets": "find_datasets", "find-models": "find_models",
    "find-collections": "find_collections", "find-all-resources": "find_all_resources",
}


def main() -> None:
    p = argparse.ArgumentParser(description="HF papers / citations research tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_limit(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--limit", type=int, help="Max results (default 10, max 50).")

    def add_arxiv(sp: argparse.ArgumentParser, required: bool = True) -> None:
        sp.add_argument("--arxiv-id", required=required,
                        help="arXiv paper ID, e.g. 2305.18290 (get IDs from search first).")

    def add_sort(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--sort", choices=["downloads", "likes", "trending"],
                        help="Sort order (default: downloads).")

    s = sub.add_parser("search", help="Search papers (Semantic Scholar when filters set).")
    s.add_argument("--query", required=True,
                   help="Search query (boolean ok: '\"exact phrase\" a | b').")
    s.add_argument("--date-from", help="Start date YYYY-MM-DD (uses Semantic Scholar).")
    s.add_argument("--date-to", help="End date YYYY-MM-DD (uses Semantic Scholar).")
    s.add_argument("--categories",
                   help="Field-of-study filter, e.g. 'Computer Science' (uses Semantic Scholar).")
    s.add_argument("--min-citations", type=int,
                   help="Minimum citation count (uses Semantic Scholar).")
    s.add_argument("--sort-by", choices=["relevance", "citationCount", "publicationDate"],
                   help="Sort order (default: relevance).")
    add_limit(s)

    t = sub.add_parser("trending", help="Trending / recent papers.")
    t.add_argument("--query", help="Optional keyword filter.")
    t.add_argument("--date", help="Date YYYY-MM-DD; defaults to recent.")
    add_limit(t)

    d = sub.add_parser("details", help="Paper metadata, citation counts, TL;DR.")
    add_arxiv(d)

    rd = sub.add_parser("read", help="Read a paper's full text (TOC first, then a section).")
    add_arxiv(rd)
    rd.add_argument("--section",
                    help="Section name or number (e.g. '3', 'Experiments'); omit for abstract + TOC.")

    cg = sub.add_parser("citation-graph", help="References and citations for a paper.")
    add_arxiv(cg)
    cg.add_argument("--direction", choices=["citations", "references", "both"],
                    help="Which edges to fetch (default: both).")
    add_limit(cg)

    ss = sub.add_parser("snippet-search", help="Semantic search over full-text passages.")
    ss.add_argument("--query", required=True, help="Passage search query.")
    ss.add_argument("--date-from", help="Start date YYYY-MM-DD.")
    ss.add_argument("--date-to", help="End date YYYY-MM-DD.")
    add_limit(ss)

    rc = sub.add_parser("recommend", help="Find similar papers (single or positive/negative examples).")
    add_arxiv(rc, required=False)
    rc.add_argument("--positive-ids", help="Comma-separated arXiv IDs to seed recommendations.")
    rc.add_argument("--negative-ids", help="Comma-separated arXiv IDs as negative examples.")
    add_limit(rc)

    fd = sub.add_parser("find-datasets", help="Datasets linked to a paper.")
    add_arxiv(fd)
    add_sort(fd)
    add_limit(fd)

    fm = sub.add_parser("find-models", help="Models linked to a paper.")
    add_arxiv(fm)
    add_sort(fm)
    add_limit(fm)

    fc = sub.add_parser("find-collections", help="Collections that include a paper.")
    add_arxiv(fc)
    add_limit(fc)

    fa = sub.add_parser("find-all-resources", help="Datasets + models + collections for a paper.")
    add_arxiv(fa)
    add_limit(fa)

    a = p.parse_args()

    args = {"operation": OPS[a.cmd]}
    for key in ("query", "arxiv_id", "section", "direction", "date", "date_from",
                "date_to", "categories", "min_citations", "sort_by",
                "positive_ids", "negative_ids", "sort", "limit"):
        val = getattr(a, key, None)
        if val is not None:
            args[key] = val
    cli.run(hf_papers_handler(args))


if __name__ == "__main__":
    main()
