#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.30", "httpx", "python-dotenv"]
# ///
"""HF Jobs CLI (submit / monitor cloud GPU jobs). Adapted from ml-intern's
hf_jobs tool. A LOCAL --script file is read and submitted INLINE, because the
job runs in a fresh container where local paths do not exist.

`run` submits real compute and costs namespace credits: confirm with the user
before invoking (per SKILL.md).

Examples:
  hf_jobs.py run --script ./train.py --hardware-flavor t4-small --timeout 1h --env MAX_STEPS=5
  hf_jobs.py logs --job-id <id>
  hf_jobs.py inspect --job-id <id>
  hf_jobs.py list --all
  hf_jobs.py cancel --job-id <id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlhelpers import cli  # noqa: E402

cli.load_env()
from mlhelpers.jobs import JobsRunner  # noqa: E402


def _kv(pairs):
    """Parse ['K=V', ...] into a dict."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="HF Jobs: submit and monitor cloud jobs")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Submit a job")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", help="Local file path; read and submitted inline")
    src.add_argument("--script-url", help="Public/raw URL to a script")
    src.add_argument("--script-inline", action="store_true", help="Read source from stdin")
    r.add_argument("--dep", action="append", help="Extra dependency (repeatable)")
    r.add_argument("--python", help="Python version for the job, e.g. 3.11.")
    r.add_argument("--script-arg", action="append", help="Arg passed to the script (repeatable)")
    r.add_argument("--image", help="Docker image to run in (default: a uv base image).")
    r.add_argument("--hardware-flavor", default="cpu-basic",
                   help="GPU/CPU flavor, e.g. a10g-large, a100-large (default: cpu-basic).")
    r.add_argument("--timeout", default="30m",
                   help="Max job duration, e.g. 2h (default: 30m; raise for real training).")
    r.add_argument("--env", action="append", help="K=V env var (repeatable)")
    r.add_argument("--secret", action="append", help="K=V secret (repeatable)")
    r.add_argument("--namespace", help="Billing namespace (default: your account).")
    r.add_argument("--no-wait", action="store_true", help="Return after submit; don't stream logs")

    for name in ("logs", "inspect", "cancel"):
        sp = sub.add_parser(name, help=f"{name.capitalize()} a job by ID")
        sp.add_argument("--job-id", required=True, help="Job ID from `run` or `list`.")
        sp.add_argument("--namespace", help="Billing namespace (default: your account).")

    lp = sub.add_parser("list", help="List jobs (running by default)")
    lp.add_argument("--all", action="store_true", help="Include finished jobs, not just running.")
    lp.add_argument("--status", help="Filter by status, e.g. RUNNING, COMPLETED, ERROR.")
    lp.add_argument("--namespace", help="Billing namespace (default: your account).")

    a = p.parse_args()
    runner = JobsRunner(token=cli.resolve_token())

    if a.cmd == "run":
        if a.script_url:
            script = a.script_url
        elif a.script:
            script = Path(a.script).read_text(encoding="utf-8")
            if "\n" not in script:
                script += "\n"  # force inline (vs. treated as a path)
        else:
            script = sys.stdin.read()
            if "\n" not in script:
                script += "\n"
        return runner.run(
            script=script,
            deps=a.dep,
            python=a.python,
            script_args=a.script_arg,
            image=a.image,
            flavor=a.hardware_flavor,
            timeout=a.timeout,
            env=_kv(a.env),
            secrets=_kv(a.secret),
            namespace=a.namespace,
            wait=not a.no_wait,
        )
    if a.cmd == "logs":
        return runner.logs(a.job_id, a.namespace)
    if a.cmd == "inspect":
        return runner.inspect(a.job_id, a.namespace)
    if a.cmd == "cancel":
        return runner.cancel(a.job_id, a.namespace)
    if a.cmd == "list":
        return runner.ps(a.namespace, show_all=a.all, status=a.status)
    return 2


if __name__ == "__main__":
    sys.exit(main())
