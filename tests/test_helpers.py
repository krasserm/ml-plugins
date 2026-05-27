"""Deterministic unit tests for the ml-research helper logic (no network).

Run from the repo root:
  uv run --with pytest --with httpx --with huggingface_hub pytest tests/ -q
"""

import base64
import importlib.util
from pathlib import Path

import pytest

from mlhelpers import cli, dataset_tools as dt, hf_access, jobs

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


# --------------------------------------------------------------------------
# jobs.py — command building
# --------------------------------------------------------------------------
def test_resolve_uv_command_url():
    cmd = jobs._resolve_uv_command("https://example.com/t.py", ["torch"])
    assert cmd == ["uv", "run", "--with", "torch", "https://example.com/t.py"]


def test_resolve_uv_command_path():
    # single token, no newline -> treated as a path in the job container
    assert jobs._resolve_uv_command("train.py") == ["uv", "run", "train.py"]


def test_resolve_uv_command_inline_roundtrips():
    script = "import os\nprint('hi')\n"
    cmd = jobs._resolve_uv_command(script, ["hf-transfer"])
    assert cmd[:2] == ["/bin/sh", "-lc"]
    # the base64 blob in the shell command must decode back to the script
    shell = cmd[2]
    blob = shell.split('echo "', 1)[1].split('"', 1)[0]
    assert base64.b64decode(blob).decode() == script
    assert "uv run --with hf-transfer -" in shell


def test_build_uv_command_ordering():
    cmd = jobs._build_uv_command("s.py", ["a", "b"], python="3.11", script_args=["--x", "1"])
    assert cmd == ["uv", "run", "--with", "a", "--with", "b", "-p", "3.11", "s.py", "--x", "1"]


def test_ensure_hf_transfer_dependency():
    assert jobs._ensure_hf_transfer_dependency(None) == ["hf-transfer"]
    assert jobs._ensure_hf_transfer_dependency(["torch"]) == ["torch", "hf-transfer"]
    # idempotent
    assert jobs._ensure_hf_transfer_dependency(["hf-transfer"]) == ["hf-transfer"]


# --------------------------------------------------------------------------
# jobs.py — env / secrets
# --------------------------------------------------------------------------
def test_add_default_env_user_overrides():
    env = jobs._add_default_env({"TQDM_DISABLE": "0", "FOO": "bar"})
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"  # default kept
    assert env["TQDM_DISABLE"] == "0"  # user override wins
    assert env["FOO"] == "bar"


def test_add_environment_variables_injects_token():
    out = jobs._add_environment_variables({"X": "1"}, "tok123")
    assert out["X"] == "1"
    assert out["HF_TOKEN"] == "tok123"
    assert out["HUGGINGFACE_HUB_TOKEN"] == "tok123"


def test_add_environment_variables_ignores_literal_dollar_token():
    out = jobs._add_environment_variables({"HF_TOKEN": "$HF_TOKEN"}, "real")
    assert out["HF_TOKEN"] == "real"  # the literal "$HF_TOKEN" was dropped


def test_add_environment_variables_no_token():
    out = jobs._add_environment_variables({"A": "b"}, None)
    assert out == {"A": "b"}  # no token keys added


# --------------------------------------------------------------------------
# jobs.py — log post-processing
# --------------------------------------------------------------------------
def test_filter_uv_install_output_truncates():
    logs = ["resolving", "downloading torch", "Installed 42 packages in 1.2s", "training step 1"]
    out = jobs._filter_uv_install_output(logs)
    assert out[0] == "[installs truncated]"
    assert "Installed 42 packages in 1.2s" in out
    assert "training step 1" in out
    assert "downloading torch" not in out


def test_strip_ansi():
    assert jobs._strip_ansi("\x1b[31mred\x1b[0m text") == "red text"


# --------------------------------------------------------------------------
# hf_access.py — namespace + billing
# --------------------------------------------------------------------------
def test_jobs_access_from_whoami():
    access = hf_access.jobs_access_from_whoami(
        {"name": "alice", "orgs": [{"name": "acme"}, {"name": "globex"}]}
    )
    assert access.username == "alice"
    assert access.default_namespace == "alice"
    assert set(access.eligible_namespaces) == {"alice", "acme", "globex"}


@pytest.mark.parametrize("msg", ["HTTP 402 Payment Required", "insufficient credits", "Please add credits"])
def test_is_billing_error_true(msg):
    assert hf_access.is_billing_error(msg)


@pytest.mark.parametrize("msg", ["", "some other 500 error", "model not found"])
def test_is_billing_error_false(msg):
    assert not hf_access.is_billing_error(msg)


# --------------------------------------------------------------------------
# jobs.JobsRunner.resolve_namespace (fake whoami, no network)
# --------------------------------------------------------------------------
class _FakeApi:
    def __init__(self, whoami):
        self._w = whoami

    def whoami(self):
        return self._w


def test_resolve_namespace_default(monkeypatch):
    r = jobs.JobsRunner(token="x")
    r.api = _FakeApi({"name": "bob", "orgs": []})
    assert r.resolve_namespace() == "bob"


def test_resolve_namespace_requested_valid():
    r = jobs.JobsRunner(token="x")
    r.api = _FakeApi({"name": "bob", "orgs": [{"name": "acme"}]})
    assert r.resolve_namespace("acme") == "acme"


def test_resolve_namespace_requested_invalid():
    r = jobs.JobsRunner(token="x")
    r.api = _FakeApi({"name": "bob", "orgs": [{"name": "acme"}]})
    with pytest.raises(hf_access.JobsAccessError):
        r.resolve_namespace("someone-else")


# --------------------------------------------------------------------------
# cli.py — token file + session
# --------------------------------------------------------------------------
def test_read_token_file(tmp_path, monkeypatch):
    tok = tmp_path / "token"
    tok.write_text("  filetok\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(tok))
    assert cli._read_token_file() == "filetok"


def test_read_token_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "nope"))
    assert cli._read_token_file() is None


def test_session():
    assert cli.Session("abc").hf_token == "abc"
    assert cli.Session().hf_token is None


# --------------------------------------------------------------------------
# github.py — notebook rendering (pure, gh-CLI-free path)
# --------------------------------------------------------------------------
def _load_github():
    spec = importlib.util.spec_from_file_location("gh_cli", SCRIPTS / "github.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_to_text():
    gh = _load_github()
    nb = (
        '{"cells": ['
        '{"cell_type": "markdown", "source": ["# Title"]},'
        '{"cell_type": "code", "source": ["print(1)\\n", "print(2)"]},'
        '{"cell_type": "code", "source": ["   "]}'  # blank -> skipped
        ']}'
    )
    out = gh._notebook_to_text(nb, "o/r", "n.ipynb")
    assert "# Title" in out
    assert "```python\nprint(1)\nprint(2)\n```" in out
    assert out.count("```python") == 1  # blank code cell skipped


def test_notebook_to_text_bad_json():
    gh = _load_github()
    assert "could not parse" in gh._notebook_to_text("{not json", "o/r", "n.ipynb")


# --------------------------------------------------------------------------
# dataset_tools.py — feature-type formatting
# --------------------------------------------------------------------------


def test_get_type_str_value_and_classlabel():
    assert dt._get_type_str({"dtype": "string", "_type": "Value"}) == "string"
    assert dt._get_type_str({"dtype": "int64", "_type": "Value"}) == "int64"
    cl = dt._get_type_str({"_type": "ClassLabel", "names": ["neg", "pos"]})
    assert cl == "ClassLabel (neg=0, pos=1)"


def test_get_type_str_list_feature_does_not_crash():
    """Regression: an SFT `messages` column arrives as a list of structs.

    Previously `_get_type_str` called `.get` on the list and raised
    `'list' object has no attribute 'get'`, which dropped the whole schema.
    """
    messages_feature = [
        {"content": {"dtype": "string", "_type": "Value"},
         "role": {"dtype": "string", "_type": "Value"}}
    ]
    out = dt._get_type_str(messages_feature)
    assert out.startswith("list<")
    assert "content" in out and "role" in out


def test_get_type_str_sequence_and_unknown():
    seq = dt._get_type_str({"_type": "Sequence", "feature": {"dtype": "float32", "_type": "Value"}})
    assert seq == "list<float32>"
    assert dt._get_type_str("weird") == "unknown"


def test_format_schema_includes_list_column():
    """The full schema table renders even when a column is a list feature."""
    info = {"dataset_info": {"features": {
        "source": {"dtype": "string", "_type": "Value"},
        "messages": [
            {"content": {"dtype": "string", "_type": "Value"},
             "role": {"dtype": "string", "_type": "Value"}}
        ],
        "num_turns": {"dtype": "int64", "_type": "Value"},
    }}}
    table = dt._format_schema(info, "default")
    assert "## Schema (default)" in table
    assert "| messages |" in table
    assert "| source |" in table
    assert "| num_turns |" in table
