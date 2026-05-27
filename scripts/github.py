#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""GitHub code-research CLI. Uses the `gh` CLI exclusively for both auth and
transport (run `gh auth login` once). No GITHUB_TOKEN is read or needed.

Examples:
  github.py find-examples --repo trl --keyword sft
  github.py list-repos --owner huggingface --sort stars
  github.py read-file --repo huggingface/trl --path examples/scripts/sft.py
"""

import argparse
import base64
import json
import shutil
import subprocess
import sys


def _gh(args, timeout=60):
    """Run a `gh` command and return stdout, or exit non-zero with the error."""
    if not shutil.which("gh"):
        print("Error: `gh` CLI not found. Install it and run `gh auth login`.")
        sys.exit(1)
    try:
        proc = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"Error: gh invocation failed: {e}")
        sys.exit(1)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "gh auth login" in err or "authentication" in err.lower():
            err += "\n(Run `gh auth login` to authenticate.)"
        print(f"Error: gh {' '.join(args)} failed:\n{err}")
        sys.exit(1)
    return proc.stdout


def find_examples(repo, keyword, org, max_results):
    full = repo if "/" in repo else f"{org}/{repo}"
    out = _gh([
        "api", f"repos/{full}/git/trees/HEAD?recursive=1",
        "--jq", '.tree[] | select(.type=="blob") | .path',
    ])
    paths = [p for p in out.splitlines() if p.strip()]
    kw = (keyword or "").lower()
    matches = [p for p in paths if kw in p.lower()] if kw else paths

    def rank(p):
        pl = p.lower()
        score = 0
        if "example" in pl:
            score -= 2
        if "/scripts/" in pl or pl.startswith("scripts/"):
            score -= 1
        return (score, len(p))

    matches.sort(key=rank)
    shown = matches[:max_results]
    if not shown:
        print(f"No files in {full} matching '{keyword}'.")
        return 0
    print(f"**{len(matches)} file(s) in {full} matching '{keyword}'** "
          f"(showing {len(shown)}):\n")
    for p in shown:
        print(f"- {p}")
        print(f"  https://github.com/{full}/blob/HEAD/{p}")
        print(f"  read: github.py read-file --repo {full} --path {p}")
    return 0


def read_file(repo, path, ref, line_start, line_end):
    q = f"repos/{repo}/contents/{path}"
    if ref:
        q += f"?ref={ref}"
    raw = _gh(["api", q, "--jq", ".content"]).strip()
    if not raw:
        print(f"No inline content for {repo}/{path} (file may exceed the 1MB "
              "contents-API limit or be binary).")
        return 1
    try:
        content = base64.b64decode(raw).decode("utf-8", "replace")
    except Exception as e:
        print(f"Could not decode {repo}/{path}: {e}")
        return 1

    if path.endswith(".ipynb"):
        print(_notebook_to_text(content, repo, path))
        return 0

    lines = content.splitlines()
    total = len(lines)
    if line_start or line_end:
        s = (line_start or 1) - 1
        e = line_end or total
        lines = lines[s:e]
        header = f"**{repo}/{path}** (lines {(line_start or 1)}-{min(e, total)} of {total})"
    else:
        header = f"**{repo}/{path}** ({total} lines)"
    print(header + "\n```\n" + "\n".join(lines) + "\n```")
    return 0


def _notebook_to_text(content, repo, path):
    """Render an .ipynb as markdown + fenced code, without nbconvert."""
    try:
        nb = json.loads(content)
    except json.JSONDecodeError:
        return f"**{repo}/{path}** (could not parse notebook JSON)"
    out = [f"**{repo}/{path}** (notebook, {len(nb.get('cells', []))} cells)\n"]
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = src.strip()
        if not src:
            continue
        if cell.get("cell_type") == "code":
            out.append("```python\n" + src + "\n```")
        else:
            out.append(src)
    return "\n\n".join(out)


def list_repos(owner, sort, limit):
    out = _gh([
        "repo", "list", owner, "--limit", str(limit),
        "--json", "name,description,stargazerCount,updatedAt,createdAt,isFork,url",
    ])
    repos = json.loads(out)
    keymap = {
        "stars": lambda r: -r.get("stargazerCount", 0),
        "updated": lambda r: r.get("updatedAt", ""),
        "created": lambda r: r.get("createdAt", ""),
        "name": lambda r: r.get("name", ""),
    }
    if sort in ("updated", "created"):
        repos.sort(key=keymap[sort], reverse=True)
    elif sort:
        repos.sort(key=keymap[sort])
    if not repos:
        print(f"No repos found for {owner}.")
        return 0
    print(f"**{len(repos)} repo(s) for {owner}:**\n")
    for r in repos:
        star = r.get("stargazerCount", 0)
        desc = (r.get("description") or "").strip()
        fork = " [fork]" if r.get("isFork") else ""
        print(f"- {r['name']} (★{star}){fork} — {desc}")
        print(f"  {r.get('url', '')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="GitHub code research via the gh CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    fe = sub.add_parser("find-examples", help="Find files in a repo by keyword")
    fe.add_argument("--repo", required=True, help="Repo name, e.g. trl (combined with --org).")
    fe.add_argument("--keyword", help="Filename keyword to match, e.g. sft.")
    fe.add_argument("--org", default="huggingface", help="GitHub org/owner (default: huggingface).")
    fe.add_argument("--max-results", type=int, default=10, help="Max files to return (default 10).")

    lr = sub.add_parser("list-repos", help="List an owner's repos")
    lr.add_argument("--owner", required=True, help="GitHub org/owner to list repos for.")
    lr.add_argument("--sort", choices=["stars", "updated", "created", "name"], default="stars",
                    help="Sort order (default: stars).")
    lr.add_argument("--limit", type=int, default=30, help="Max repos to return (default 30).")

    rf = sub.add_parser("read-file", help="Read a file from a repo")
    rf.add_argument("--repo", required=True, help="owner/repo")
    rf.add_argument("--path", required=True, help="File path within the repo.")
    rf.add_argument("--ref", help="Branch, tag, or commit (default: the repo's default branch).")
    rf.add_argument("--line-start", type=int, help="First line to return (1-based).")
    rf.add_argument("--line-end", type=int, help="Last line to return (inclusive).")

    a = p.parse_args()
    if a.cmd == "find-examples":
        return find_examples(a.repo, a.keyword, a.org, a.max_results)
    if a.cmd == "list-repos":
        return list_repos(a.owner, a.sort, a.limit)
    if a.cmd == "read-file":
        return read_file(a.repo, a.path, a.ref, a.line_start, a.line_end)
    return 2


if __name__ == "__main__":
    sys.exit(main())
