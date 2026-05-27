#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""PreToolUse hook: enforce the program.md budget envelope during autonomous runs.

Registered (via the plugin's hooks/hooks.json) to run on every Bash call; it
self-filters and acts only on:
  - `hf_jobs.py run`                     -> budget: max_jobs, max_walltime,
                                            allowed_flavors, max_timeout
  - `hf_repo.py files upload` / `delete` -> scope:  output_repos, allow_deletes

Decision (PreToolUse contract):
  - No program.md envelope present  -> DEFER (exit 0, no output): the hook stays
                                       out of interactive mode entirely. Spend/
                                       write confirmation there is skill-driven
                                       (see ml-research-task's Authorization).
  - Within the envelope             -> "allow"  (bypass the permission prompt).
  - Breach / malformed envelope     -> "deny"   (escalates to the user / halts loop).

Reads the PreToolUse stdin JSON (`cwd`, `tool_input.command`) and tallies prior
usage from runs/ledger.jsonl. The envelope is the fenced ```yaml block under the
"## Budget envelope" heading in program.md (see docs/program.template.md).
"""

import json
import os
import re
import shlex
import sys
from fnmatch import fnmatch
from pathlib import Path

DUR_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def emit(decision: str, reason: str) -> None:
    """Emit a PreToolUse permission decision and exit."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def defer() -> None:
    """No opinion: exit 0 with no output so the normal permission flow applies."""
    sys.exit(0)


def parse_duration(s: object) -> int:
    """'30m', '1h', '2h30m', '3600', '1d' -> seconds. 0 if unparseable/empty."""
    text = str(s).strip().lower()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    total, matched = 0, False
    for num, unit in re.findall(r"(\d+)\s*([smhd])", text):
        matched = True
        total += int(num) * DUR_UNITS[unit]
    return total if matched else 0


def _project_dirs(cwd: Path) -> list[Path]:
    dirs = [cwd]
    proj = os.environ.get("CLAUDE_PROJECT_DIR")
    if proj:
        dirs.append(Path(proj))
    return dirs


def find_envelope(cwd: Path) -> dict | None:
    """Parsed budget envelope, or None if no program.md / no envelope section.

    Raises ValueError if an envelope section exists but cannot be parsed.
    """
    for d in _project_dirs(cwd):
        p = d / "program.md"
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        m = re.search(r"##\s+Budget envelope.*?```ya?ml\s*(.*?)```", text, re.S | re.I)
        if not m:
            return None  # program.md without an envelope => treat as interactive
        import yaml  # lazy: keeps the pure decision logic importable without pyyaml
        try:
            env = yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            raise ValueError(f"could not parse Budget envelope YAML: {e}")
        if not isinstance(env, dict):
            raise ValueError("Budget envelope is not a YAML mapping")
        return env
    return None


def ledger_rows(cwd: Path) -> list[dict]:
    for d in _project_dirs(cwd):
        lp = d / "runs" / "ledger.jsonl"
        if not lp.is_file():
            continue
        rows = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows
    return []


def arg_value(tokens: list[str], flag: str) -> str | None:
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def script_and_op(tokens: list[str]) -> tuple[str | None, str | None]:
    """('hf_jobs.py', 'run') etc. from `uv run $CLAUDE_PLUGIN_ROOT/scripts/X.py OP ...`."""
    for i, t in enumerate(tokens):
        if t.endswith(".py"):
            op = tokens[i + 1] if i + 1 < len(tokens) else None
            return Path(t).name, op
    return None, None


def check_jobs(env: dict, tokens: list[str], cwd: Path) -> None:
    flavor = arg_value(tokens, "--hardware-flavor") or "cpu-basic"
    timeout_s = parse_duration(arg_value(tokens, "--timeout") or "30m")

    job_rows = [r for r in ledger_rows(cwd) if r.get("job_id")]
    jobs_used = len(job_rows)
    walltime_used = sum(parse_duration(r.get("timeout", 0)) for r in job_rows)

    max_jobs = env.get("max_jobs")
    allowed = env.get("allowed_flavors") or []
    max_timeout_s = parse_duration(env.get("max_timeout", 0))
    max_walltime_s = parse_duration(env.get("max_walltime", 0))

    violations: list[str] = []
    if isinstance(max_jobs, int) and jobs_used + 1 > max_jobs:
        violations.append(f"max_jobs={max_jobs} reached ({jobs_used} already submitted)")
    if allowed and flavor not in allowed:
        violations.append(f"hardware-flavor '{flavor}' not in allowed_flavors {allowed}")
    if max_timeout_s and timeout_s > max_timeout_s:
        violations.append(f"--timeout {timeout_s}s exceeds max_timeout {max_timeout_s}s")
    if max_walltime_s and walltime_used + timeout_s > max_walltime_s:
        violations.append(
            f"walltime budget exceeded: {walltime_used + timeout_s}s > max_walltime {max_walltime_s}s"
        )

    if violations:
        emit("deny", "Budget envelope exceeded — " + "; ".join(violations)
             + ". Stop the loop and report to the user.")
    emit("allow", f"Within budget envelope (job {jobs_used + 1}"
         + (f"/{max_jobs}" if isinstance(max_jobs, int) else "") + f", flavor {flavor}).")


def _repo_sub_op(tokens: list[str]) -> str | None:
    """The hf_repo.py `files` sub-operation (list|read|upload|delete), if present."""
    for i, t in enumerate(tokens):
        if t == "files" and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def check_repo(env: dict, tokens: list[str]) -> None:
    # tokens: uv run $CLAUDE_PLUGIN_ROOT/scripts/hf_repo.py files <op> --repo-id ...
    sub_op = _repo_sub_op(tokens)
    if sub_op not in ("upload", "delete"):
        defer()  # list/read are read-only

    if sub_op == "delete" and not env.get("allow_deletes", False):
        emit("deny", "Remote deletes are disabled by the budget envelope "
             "(allow_deletes: false). Stop and report to the user.")

    repo_id = arg_value(tokens, "--repo-id") or ""
    patterns = env.get("output_repos") or []
    if patterns and not any(fnmatch(repo_id, pat) for pat in patterns):
        emit("deny", f"repo '{repo_id}' is out of scope (output_repos {patterns}). "
             "Stop and report to the user.")
    emit("allow", f"{sub_op} to '{repo_id}' is within envelope scope.")


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        defer()

    command = (data.get("tool_input") or {}).get("command", "") or ""
    cwd = Path(data.get("cwd") or ".")

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    script, op = script_and_op(tokens)

    try:
        env = find_envelope(cwd)
    except ValueError as e:
        # Envelope present but broken: never auto-approve spend on a bad envelope.
        emit("deny", f"Autonomous run blocked: {e}. Fix program.md's Budget envelope.")

    if env is None:
        # Interactive mode (no budget envelope): the hook has no budget to
        # enforce, so it stays out of the way and defers. Confirming spend/write
        # actions is the skill's job (ml-research-task's Authorization section).
        defer()

    if script == "hf_jobs.py" and op == "run":
        check_jobs(env, tokens, cwd)
    elif script == "hf_repo.py" and op == "files":
        check_repo(env, tokens)
    else:
        defer()


if __name__ == "__main__":
    main()
