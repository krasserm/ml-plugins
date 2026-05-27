"""Slim, synchronous HF Jobs runner for the ml-intern Claude Code scaffold.

Adapted from ml-intern's agent/tools/jobs_tool.py (Apache-2.0). The pure
command-building / env / log-filtering helpers are copied verbatim; the
session/telemetry/trackio/Event machinery and the async log-bridge are removed.
Because the CLI does nothing concurrently, log streaming is a plain blocking
loop over HfApi.fetch_job_logs with reconnect-on-drop, matching the original's
retry semantics.
"""

from __future__ import annotations

import base64
import http.client
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from .hf_access import JobsAccessError, is_billing_error, jobs_access_from_whoami
from .utilities import format_job_details, format_jobs_table

# --------------------------------------------------------------------------
# Hardware flavors (for reference / validation; copied from jobs_tool.py)
# --------------------------------------------------------------------------
CPU_FLAVORS = ["cpu-basic", "cpu-upgrade"]
GPU_FLAVORS = [
    "t4-small", "t4-medium", "a10g-small", "a10g-large", "a10g-largex2",
    "a10g-largex4", "a100-large", "a100x4", "a100x8", "l4x1", "l4x4",
    "l40sx1", "l40sx4", "l40sx8",
]
SPECIALIZED_FLAVORS = ["inf2x6"]
ALL_FLAVORS = CPU_FLAVORS + GPU_FLAVORS + SPECIALIZED_FLAVORS

UV_DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"

_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "ERROR"}


# --------------------------------------------------------------------------
# Pure helpers (verbatim from jobs_tool.py)
# --------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _filter_uv_install_output(logs: List[str]) -> List[str]:
    """Collapse UV package-install spam to a single '[installs truncated]' line."""
    if not logs:
        return logs
    install_pattern = re.compile(
        r"^Installed\s+\d+\s+packages?\s+in\s+\d+(?:\.\d+)?\s*(?:ms|s)$"
    )
    install_line_idx = None
    for idx, line in enumerate(logs):
        if install_pattern.match(line.strip()):
            install_line_idx = idx
            break
    if install_line_idx is not None and install_line_idx > 0:
        return ["[installs truncated]"] + logs[install_line_idx:]
    return logs


_DEFAULT_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "UV_NO_PROGRESS": "1",
}


def _add_default_env(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(_DEFAULT_ENV)
    result.update(params or {})
    return result


def _add_environment_variables(
    params: Optional[Dict[str, Any]], user_token: Optional[str] = None
) -> Dict[str, Any]:
    token = user_token or ""
    result = dict(params or {})
    if result.get("HF_TOKEN", "").strip().startswith("$"):
        result.pop("HF_TOKEN", None)
    if token:
        result["HF_TOKEN"] = token
        result["HUGGINGFACE_HUB_TOKEN"] = token
    return result


def _build_uv_command(
    script: str,
    with_deps: Optional[List[str]] = None,
    python: Optional[str] = None,
    script_args: Optional[List[str]] = None,
) -> List[str]:
    parts = ["uv", "run"]
    if with_deps:
        for dep in with_deps:
            parts.extend(["--with", dep])
    if python:
        parts.extend(["-p", python])
    parts.append(script)
    if script_args:
        parts.extend(script_args)
    return parts


def _wrap_inline_script(
    script: str,
    with_deps: Optional[List[str]] = None,
    python: Optional[str] = None,
    script_args: Optional[List[str]] = None,
) -> str:
    encoded = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    uv_command = _build_uv_command("-", with_deps, python, script_args)
    return f'echo "{encoded}" | base64 -d | {" ".join(uv_command)}'


def _ensure_hf_transfer_dependency(deps: Optional[List[str]]) -> List[str]:
    if isinstance(deps, list):
        deps_copy = deps.copy()
        if "hf-transfer" not in deps_copy:
            deps_copy.append("hf-transfer")
        return deps_copy
    return ["hf-transfer"]


def _resolve_uv_command(
    script: str,
    with_deps: Optional[List[str]] = None,
    python: Optional[str] = None,
    script_args: Optional[List[str]] = None,
) -> List[str]:
    """URL -> uv run <url>; multi-line content -> inline base64 over stdin;
    single token -> uv run <path> (only valid if the path exists in the job)."""
    if script.startswith("http://") or script.startswith("https://"):
        return _build_uv_command(script, with_deps, python, script_args)
    if "\n" in script:
        wrapped = _wrap_inline_script(script, with_deps, python, script_args)
        return ["/bin/sh", "-lc", wrapped]
    return _build_uv_command(script, with_deps, python, script_args)


def _job_info_to_dict(job_info) -> Dict[str, Any]:
    return {
        "id": job_info.id,
        "status": {"stage": job_info.status.stage, "message": job_info.status.message},
        "command": job_info.command,
        "createdAt": job_info.created_at.isoformat(),
        "dockerImage": job_info.docker_image,
        "spaceId": job_info.space_id,
        "hardware_flavor": job_info.flavor,
        "owner": {"name": job_info.owner.name},
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
class JobsRunner:
    """Synchronous wrapper over the HfApi Jobs surface."""

    def __init__(self, token: Optional[str], namespace: Optional[str] = None):
        self.token = token
        self.api = HfApi(token=token)
        self._namespace = namespace

    # -- namespace resolution (sync; mirrors hf_access.resolve_jobs_namespace) --
    def resolve_namespace(self, requested: Optional[str] = None) -> str:
        requested = requested or self._namespace
        try:
            whoami = self.api.whoami()
        except Exception as e:
            if requested:
                self._namespace = requested
                return requested
            raise JobsAccessError(f"Could not resolve HF namespace: {e}")
        access = jobs_access_from_whoami(whoami)
        if requested:
            if not access.eligible_namespaces or requested in access.eligible_namespaces:
                self._namespace = requested
                return requested
            raise JobsAccessError(
                "You can only run jobs under your own account or an org you belong to. "
                f"Allowed: {', '.join(access.eligible_namespaces) or '(none)'}"
            )
        if access.default_namespace:
            self._namespace = access.default_namespace
            return access.default_namespace
        raise JobsAccessError("Couldn't resolve a Hugging Face namespace for this token.")

    # -- log streaming with reconnect (mirrors _wait_for_job_completion) --
    def _stream_until_done(self, job_id: str, namespace: str) -> str:
        printed = 0
        retry_delay = 5
        for _ in range(100):
            try:
                idx = 0
                for line in self.api.fetch_job_logs(job_id=job_id, namespace=namespace):
                    # On reconnect the stream restarts from the top; only emit
                    # lines we haven't printed yet.
                    if idx >= printed:
                        print(_strip_ansi(line), flush=True)
                        printed = idx + 1
                    idx += 1
                break  # generator exhausted -> stream finished
            except (
                ConnectionError, TimeoutError, OSError,
                http.client.IncompleteRead,
                httpx.RemoteProtocolError, httpx.ReadError, HfHubHTTPError,
            ):
                try:
                    stage = self.api.inspect_job(job_id=job_id, namespace=namespace).status.stage
                    if stage in _TERMINAL_STATES:
                        break
                except (ConnectionError, TimeoutError, OSError):
                    pass
                time.sleep(retry_delay)
                continue
        # Final status (API may lag a few seconds behind the stream ending).
        final = "UNKNOWN"
        for _ in range(6):
            final = self.api.inspect_job(job_id=job_id, namespace=namespace).status.stage
            if final in _TERMINAL_STATES:
                break
            time.sleep(2.5)
        return final

    def run(
        self,
        *,
        script: str,
        deps: Optional[List[str]] = None,
        python: Optional[str] = None,
        script_args: Optional[List[str]] = None,
        image: Optional[str] = None,
        flavor: str = "cpu-basic",
        timeout: str = "30m",
        env: Optional[Dict[str, str]] = None,
        secrets: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        wait: bool = True,
    ) -> int:
        """Submit a job. ``script`` is a URL, or inline source (with newlines)."""
        ns = self.resolve_namespace(namespace)
        deps = _ensure_hf_transfer_dependency(deps)
        command = _resolve_uv_command(script, deps, python, script_args)
        env_dict = _add_default_env(env)
        secrets_dict = _add_environment_variables(secrets, self.token)
        try:
            job = self.api.run_job(
                image=image or UV_DEFAULT_IMAGE,
                command=command,
                env=env_dict,
                secrets=secrets_dict,
                flavor=flavor,
                timeout=timeout,
                namespace=ns,
            )
        except HfHubHTTPError as e:
            if is_billing_error(str(e)):
                print(
                    f"Hugging Face Jobs rejected this run: namespace `{ns}` has no "
                    "available credits. HF Jobs are billed with namespace credits "
                    "(separate from HF Pro). Add credits at "
                    "https://huggingface.co/settings/billing, then re-run."
                )
                return 1
            raise
        print(f"Job submitted: {job.id}")
        print(f"View at: {job.url}")
        print(f"Namespace: {ns} | Flavor: {flavor} | Timeout: {timeout}")
        if not wait:
            return 0
        print("--- streaming logs (reconnects on drop) ---", flush=True)
        final = self._stream_until_done(job.id, ns)
        print(f"\n**Final Status:** {final}")
        return 0 if final == "COMPLETED" else 1

    def logs(self, job_id: str, namespace: Optional[str] = None) -> int:
        ns = self.resolve_namespace(namespace)
        lines = list(self.api.fetch_job_logs(job_id=job_id, namespace=ns))
        if not lines:
            print(f"No logs available for job {job_id}")
            return 0
        print(_strip_ansi("\n".join(lines)))
        return 0

    def inspect(self, job_id: str, namespace: Optional[str] = None) -> int:
        ns = self.resolve_namespace(namespace)
        info = self.api.inspect_job(job_id=job_id, namespace=ns)
        print(format_job_details([_job_info_to_dict(info)]))
        return 0

    def ps(self, namespace: Optional[str] = None, show_all: bool = False,
           status: Optional[str] = None) -> int:
        ns = self.resolve_namespace(namespace)
        jobs = list(self.api.list_jobs(namespace=ns))
        if not show_all:
            jobs = [j for j in jobs if j.status.stage == "RUNNING"]
        if status:
            jobs = [j for j in jobs if status.upper() in j.status.stage]
        if not jobs:
            print("No running jobs found (use --all to show every job)." if not show_all
                  else "No jobs found.")
            return 0
        print(format_jobs_table([_job_info_to_dict(j) for j in jobs]))
        return 0

    def cancel(self, job_id: str, namespace: Optional[str] = None) -> int:
        ns = self.resolve_namespace(namespace)
        self.api.cancel_job(job_id=job_id, namespace=ns)
        print(f"Job {job_id} cancelled.")
        return 0
