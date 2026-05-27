#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "python-dotenv"]
# ///
"""HF repo file operations CLI. Wraps ml-intern's hf_repo_files_handler.
upload/delete write to the Hub: confirm with the user first (per SKILL.md).

Examples:
  hf_repo.py files list --repo-id gpt2
  hf_repo.py files read --repo-id gpt2 --path config.json
  hf_repo.py files upload --repo-id me/model --path train.py --content-file ./train.py
  hf_repo.py files delete --repo-id me/model --patterns "*.tmp" "logs/"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlhelpers import cli  # noqa: E402

cli.load_env()
from mlhelpers.hf_repo_files_tool import hf_repo_files_handler  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="HF repo file operations")
    sub = p.add_subparsers(dest="group", required=True)
    files = sub.add_parser("files", help="File operations on a repo")
    files.add_argument("op", choices=["list", "read", "upload", "delete"],
                       help="list/read are read-only; upload/delete write to the Hub.")
    files.add_argument("--repo-id", required=True, help="Repo ID, e.g. gpt2 or me/model.")
    files.add_argument("--repo-type", choices=["model", "dataset", "space"],
                       help="Repo type (default: model).")
    files.add_argument("--revision", help="Branch, tag, or commit (default: main).")
    files.add_argument("--path", help="File path within the repo (read/upload).")
    files.add_argument("--content-file", help="Local file whose contents to upload")
    files.add_argument("--patterns", nargs="+", help="Delete patterns/paths")
    files.add_argument("--create-pr", action="store_true",
                       help="Open a PR instead of committing directly (upload/delete).")
    files.add_argument("--commit-message", help="Commit message for upload/delete.")
    a = p.parse_args()

    args = {"operation": a.op, "repo_id": a.repo_id}
    if a.repo_type is not None:
        args["repo_type"] = a.repo_type
    if a.revision is not None:
        args["revision"] = a.revision
    if a.path is not None:
        args["path"] = a.path
    if a.patterns is not None:
        args["patterns"] = a.patterns
    if a.create_pr:
        args["create_pr"] = True
    if a.commit_message is not None:
        args["commit_message"] = a.commit_message
    if a.op == "upload":
        if not a.content_file:
            p.error("upload requires --content-file")
        args["content"] = Path(a.content_file).read_text(encoding="utf-8")

    cli.run(hf_repo_files_handler(args, session=cli.hf_session()))


if __name__ == "__main__":
    main()
