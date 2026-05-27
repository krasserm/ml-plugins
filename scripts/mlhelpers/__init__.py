"""mlhelpers: HF-ecosystem tool logic for the ml-intern Claude Code scaffold.

Most modules in this package are copied/adapted (with minor import fixups) from
HuggingFace's ml-intern (https://github.com/huggingface/ml-intern, Apache-2.0).
They are the pure API/IO logic behind ml-intern's tools; the surrounding agent
loop, LLM calls, and session machinery are replaced by Claude Code itself.

Keep __init__ import-light: each CLI imports only the submodule it needs so that
its PEP-723 dependency set stays minimal.
"""
