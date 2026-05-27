# Idea: migrate ml-intern into a Claude Code scaffold

This describes the *idea* of the migration so another agent can carry it out
itself. It is anchored on Hugging Face's
[`ml-intern`](https://github.com/huggingface/ml-intern) as the transformation
source. It deliberately omits implementation details — the *how* is for the
agent doing the work to decide. (`ml-research` is one realisation of this idea.)

## The source: ml-intern

ml-intern (Apache-2.0) is an autonomous ML engineering agent shipped as a
standalone Python program. Its heart is a hand-written agentic loop
(`agent/core/agent_loop.py`): each turn it calls an LLM via litellm, parses the
tool calls the model returns, runs them through a tool router, appends the
results, and repeats. Around that loop it builds its own context compaction, a
planning tool, local bash/read/write/edit tools, web search, approval prompts,
multi-provider model switching, trace upload, and a web UI.

Its capabilities live in ~15 tools (`agent/tools/`). Most are thin wrappers over
external services:

- **papers** — arXiv + HF + Semantic Scholar (search, citation graphs, read
  paper sections, find datasets/models),
- **docs** — explore/fetch HF library docs and the REST API reference,
- **dataset inspection** — schema, splits, sample rows,
- **github** — find and read example code,
- **HF repo file operations**, and **HF Jobs** (submit/monitor cloud GPU jobs);
  plus an HF Space GPU sandbox.

One tool is different: **research** is itself a small sub-agent — it runs its own
LLM loop with a read-only subset of the tools and returns a summary, so research
doesn't fill the main context.

What makes ml-intern good is concentrated in its system prompt
(`agent/prompts/system_prompt_v3.yaml`): a research-first ML playbook — start
from the literature, validate datasets before training, preflight before
launching GPU jobs, recover from errors without quietly changing scope.

## The insight

Almost everything ml-intern builds by hand — the loop, compaction, planning,
local file/shell tools, web search, approvals — is something **Claude Code
already provides**. ml-intern only re-implements them because it is a standalone
program. Run the same idea *inside* Claude Code and that scaffolding disappears.

What remains worth keeping is the ml-intern-specific value:

1. the playbook (its system prompt),
2. the research sub-agent, and
3. the Hugging Face ecosystem tool wrappers.

## The transformation

- ml-intern's agent loop, context compaction, planning tool, local
  bash/read/write/edit, and web search → **drop them**; Claude Code is the loop
  and supplies the rest.
- `system_prompt_v3.yaml` (the playbook) → a Claude Code **skill** that loads on
  ML tasks and tells Claude how to work.
- the **research** tool (the only tool that itself calls an LLM) → a Claude Code
  **subagent** with its own context and a read-only research toolset.
- the pure API-wrapper tools (papers, docs, dataset inspect, github, HF repo,
  HF Jobs) → **tools Claude can invoke directly**, carrying over their API/IO
  logic but none of ml-intern's loop/session coupling.
- multi-provider model switching, trace upload, telemetry, the web
  frontend/backend → drop; Claude Code is the interface and the model.

## Why it's worth doing

Because Claude Code becomes the loop, every model call — main reasoning, the
research subagent, compaction — runs on a **Claude Code subscription**. The
carried-over tools authenticate only to Hugging Face (and GitHub) and contain no
model calls, so the whole thing runs with **no LLM API key**. That is the point:
ml-intern's behaviour, paid for through a subscription instead of per-token API
billing.

## Trade-offs to expect

- **Model choice** becomes whatever Claude Code is running, rather than
  ml-intern's any-provider switching.
- **Approval** in interactive mode is prompt-based (the playbook tells Claude to
  confirm before spending), which is softer than a hard gate but allows
  pre-approving a batch upfront. Autonomous runs keep a hard gate: the budget-
  envelope hook allows/denies every spend against `program.md`'s limits.
- **Peripheral features** (trace upload, telemetry, the hosted web UI) fall away
  because Claude Code is the interface.

## Definition of done

Reproduce one real ml-intern task end-to-end through the scaffold — for example,
fine-tune a model on HF Jobs: research the recipe, validate the dataset, run the
training job, and confirm a pushed model — with no LLM API key configured.

## What's left to you

Everything about *how*: the project layout, where the playbook and tool logic
live, how the tools are packaged and invoked, how Hugging Face / GitHub
credentials are resolved, how much of ml-intern's tool code to reuse versus
rewrite, which optional tools (e.g. the GPU sandbox) to include, and how
approvals are enforced. ml-intern is Apache-2.0, so its tool code may be reused
with attribution.
