"""Shared glue for the thin CLI wrappers in scripts/.

Each CLI script:
  1. inserts the scripts dir on sys.path (so `from mlhelpers import ...` works
     even though `uv run` on a PEP-723 script ignores the project venv),
  2. calls load_env() to pull HF_TOKEN / GITHUB_TOKEN from the project .env,
  3. builds an `arguments` dict matching the original ml-intern *_TOOL_SPEC,
  4. awaits the original async handler and prints its (text, ok) result.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def load_env() -> None:
    """Load the nearest .env (walking up from cwd) into os.environ."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        candidate = d / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            return
    load_dotenv()


class Session:
    """Minimal stand-in for ml-intern's session object.

    The ported read-only handlers only ever read ``session.hf_token``.
    """

    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token


def _read_token_file() -> str | None:
    """Read the hf-cli token cache directly (no huggingface_hub dependency).

    Honors HF_TOKEN_PATH, then HF_HOME/token, then ~/.cache/huggingface/token.
    """
    path = os.environ.get("HF_TOKEN_PATH")
    if not path:
        hf_home = os.environ.get("HF_HOME")
        path = (
            str(Path(hf_home) / "token")
            if hf_home
            else str(Path.home() / ".cache" / "huggingface" / "token")
        )
    try:
        token = Path(path).read_text(encoding="utf-8").strip()
        return token or None
    except OSError:
        return None


def resolve_token() -> str | None:
    """Resolve an HF token: explicit env, then huggingface_hub cache (if
    installed), then the token file read directly.

    The direct file read matters because the lighter read-only scripts do not
    depend on huggingface_hub, so its get_token() cache lookup is unavailable.
    """
    from .tokens import resolve_hf_token

    return resolve_hf_token(os.environ.get("HF_TOKEN")) or _read_token_file()


def hf_session() -> Session:
    return Session(resolve_token())


def emit(result: tuple[str, bool]) -> None:
    """Print a handler's (text, ok) result and exit with a matching code."""
    text, ok = result
    print(text if text is not None else "")
    sys.exit(0 if ok else 1)


def run(coro) -> None:
    emit(asyncio.run(coro))
