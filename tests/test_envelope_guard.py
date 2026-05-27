"""Offline tests for the envelope_guard PreToolUse hook (no network, no pyyaml).

Exercises the pure decision logic (check_jobs / check_repo / parse helpers) by
loading the hook module directly and capturing the JSON it emits. YAML parsing in
find_envelope is covered by the end-to-end hook run in the plan's verification.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "envelope_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("envelope_guard", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eg = _load()

ENV = {
    "max_jobs": 2,
    "max_walltime": "8h",
    "allowed_flavors": ["t4-small", "a10g-large"],
    "max_timeout": "2h",
    "allow_deletes": False,
    "output_repos": ["me/sweep-*"],
}


def decision(func, *args):
    """Run a check_* function (prints JSON then sys.exit) and return the decision."""
    buf = io.StringIO()
    with pytest.raises(SystemExit) as ei, contextlib.redirect_stdout(buf):
        func(*args)
    assert ei.value.code == 0
    out = buf.getvalue().strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


def write_ledger(tmp_path, rows):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def test_parse_duration():
    assert eg.parse_duration("30m") == 1800
    assert eg.parse_duration("1h") == 3600
    assert eg.parse_duration("2h30m") == 9000
    assert eg.parse_duration("3600") == 3600
    assert eg.parse_duration("1d") == 86400
    assert eg.parse_duration("") == 0
    assert eg.parse_duration("garbage") == 0


def test_arg_value():
    toks = ["run", "--hardware-flavor", "t4-small", "--timeout", "1h"]
    assert eg.arg_value(toks, "--hardware-flavor") == "t4-small"
    assert eg.arg_value(toks, "--timeout") == "1h"
    assert eg.arg_value(toks, "--missing") is None
    assert eg.arg_value(["--env=A=B"], "--env") == "A=B"


def test_script_and_op():
    toks = ["uv", "run", "$CLAUDE_PLUGIN_ROOT/scripts/hf_jobs.py", "run", "--script", "x"]
    assert eg.script_and_op(toks) == ("hf_jobs.py", "run")


def test_jobs_within_budget(tmp_path):
    write_ledger(tmp_path, [])
    toks = ["hf_jobs.py", "run", "--hardware-flavor", "t4-small", "--timeout", "1h"]
    assert decision(eg.check_jobs, ENV, toks, tmp_path)["permissionDecision"] == "allow"


def test_jobs_over_max_jobs(tmp_path):
    write_ledger(tmp_path, [{"job_id": "a", "timeout": "1h"}, {"job_id": "b", "timeout": "1h"}])
    toks = ["hf_jobs.py", "run", "--hardware-flavor", "t4-small", "--timeout", "1h"]
    d = decision(eg.check_jobs, ENV, toks, tmp_path)
    assert d["permissionDecision"] == "deny"
    assert "max_jobs" in d["permissionDecisionReason"]


def test_jobs_bad_flavor(tmp_path):
    write_ledger(tmp_path, [])
    toks = ["hf_jobs.py", "run", "--hardware-flavor", "a100x8", "--timeout", "1h"]
    d = decision(eg.check_jobs, ENV, toks, tmp_path)
    assert d["permissionDecision"] == "deny"
    assert "flavor" in d["permissionDecisionReason"]


def test_jobs_over_timeout(tmp_path):
    write_ledger(tmp_path, [])
    toks = ["hf_jobs.py", "run", "--hardware-flavor", "t4-small", "--timeout", "5h"]
    assert decision(eg.check_jobs, ENV, toks, tmp_path)["permissionDecision"] == "deny"


def test_jobs_over_walltime(tmp_path):
    write_ledger(tmp_path, [{"job_id": f"j{i}", "timeout": "1h"} for i in range(8)])
    env = dict(ENV, max_jobs=100)  # so max_jobs doesn't trip first
    toks = ["hf_jobs.py", "run", "--hardware-flavor", "t4-small", "--timeout", "1h"]
    d = decision(eg.check_jobs, env, toks, tmp_path)
    assert d["permissionDecision"] == "deny"
    assert "walltime" in d["permissionDecisionReason"]


def test_repo_delete_blocked():
    toks = ["hf_repo.py", "files", "delete", "--repo-id", "me/sweep-1", "--patterns", "*.tmp"]
    assert decision(eg.check_repo, ENV, toks)["permissionDecision"] == "deny"


def test_repo_upload_out_of_scope():
    toks = ["hf_repo.py", "files", "upload", "--repo-id", "other/model", "--path", "x"]
    assert decision(eg.check_repo, ENV, toks)["permissionDecision"] == "deny"


def test_repo_upload_in_scope():
    toks = ["hf_repo.py", "files", "upload", "--repo-id", "me/sweep-3", "--path", "x"]
    assert decision(eg.check_repo, ENV, toks)["permissionDecision"] == "allow"


# --- interactive mode (no program.md): the hook defers everything -------------
# Spend/write confirmation is skill-driven there, not the hook's job, so the hook
# emits no decision (defer) for every command when no budget envelope is present.


def run_main(monkeypatch, command, cwd):
    """Drive main() with a synthetic PreToolUse payload; return decision or None."""
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    payload = json.dumps({"tool_input": {"command": command}, "cwd": str(cwd)})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    buf = io.StringIO()
    with pytest.raises(SystemExit) as ei, contextlib.redirect_stdout(buf):
        eg.main()
    assert ei.value.code == 0
    out = buf.getvalue().strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


def test_interactive_defers_everything(tmp_path, monkeypatch):
    # No program.md: the hook defers (no output) for every command, including
    # spend/write actions — confirming those is the skill's job, not the hook's.
    spend_or_write = [
        "uv run /x/scripts/hf_jobs.py run --hardware-flavor cpu-basic --timeout 10m",
        "uv run /x/scripts/hf_repo.py files upload --repo-id me/x --path y",
        "uv run /x/scripts/hf_repo.py files delete --repo-id me/x --patterns z",
    ]
    read_only = [
        "uv run /x/scripts/hf_repo.py files list --repo-id gpt2",
        "uv run /x/scripts/papers.py search --query z",
        "ls -la",
    ]
    for cmd in spend_or_write + read_only:
        assert run_main(monkeypatch, cmd, tmp_path) is None, cmd
