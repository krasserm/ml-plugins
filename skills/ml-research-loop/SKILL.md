---
name: ml-research-loop
description: >-
  Autonomous, budget-bounded loop that runs MANY ML experiments toward a goal
  with no human in the loop — hypothesize, train on HF Jobs, evaluate, keep/
  discard, repeat until the target metric or budget is hit. Requires a
  `program.md` task spec (goal, keep/discard criteria, and a budget envelope).
  Use for an unattended sweep / multi-experiment campaign. For a single task,
  use ml-research-task instead. Triggers on "run the sweep", "autonomous loop",
  "ml-research-loop", "run experiments until".
---

# ML Loop

You are the **orchestrator** of an autonomous research campaign. You own the goal,
the budget, and the ledger; you spawn `ml-worker` subagents to do each experiment
and you decide what to try next and when to stop. The per-experiment discipline
lives in `ml-research-task` (which the workers preload) — you do not run experiments
yourself. Invoke helper scripts as `uv run ${CLAUDE_PLUGIN_ROOT}/scripts/<name>.py`.

## Preconditions

The loop runs from a `program.md` at the repo root with: Goal (measurable metric
+ target), Scope (allowed models/datasets/output repos), Keep/discard criteria,
and a **Budget envelope** (the envelope is what authorizes unattended spend, via
the budget hook).

If `program.md` is missing or has no envelope, do NOT start a loop. Instead,
**author it with the user** (do not just tell them to copy the template):

1. Read the bundled `program.template.md` in this skill's directory as the
   structure to fill.
2. Pre-fill everything you can already infer from the user's request and the
   repo: the goal, a reference training script/command, the output-repo prefix.
3. Ask the user only for the pieces you cannot infer: the measurable goal +
   **target metric**, the allowed **models/datasets/output repos**, and the
   **budget envelope** values (`max_jobs`, `max_walltime`, `allowed_flavors`,
   `max_timeout`, `allow_deletes`). Propose sensible defaults from the template
   so they can just confirm.
4. Write the completed `program.md` to the repo root and show it for
   confirmation. Begin iterating only once it exists and the user has approved
   the budget (the envelope is real authorization to spend).

## State lives in the ledger, not your context

You WILL be compacted across a long campaign. Treat `runs/ledger.jsonl` +
`program.md` as the source of truth and **re-read both at the start of every
iteration**. You are the single writer of the ledger. Append one row per
experiment:

```json
{"iter": 3, "ts": "...", "hypothesis": "...", "config": {...}, "job_id": "...",
 "status": "submitted|completed|error", "metric": 0.0, "hub_url": "...",
 "decision": "keep|discard", "best_so_far": 0.0, "timeout": "1h"}
```

Keep `runs/summary.md` as a human-readable running table (best-so-far, what's been
tried). Record `timeout` on every submitted row — the budget hook sums it for the
walltime budget.

## Driver: run self-paced, wake on job completion

Run this loop self-paced (e.g. under `/loop` with no interval). HF Jobs are async
and take minutes to hours, so do not block: submit with `--no-wait`, then use
**ScheduleWakeup** with a delay keyed to the expected job duration to yield until
the job should be done. On each wake, re-read state and advance one step.

## Iteration protocol

1. **Assess** — re-read `program.md` + ledger; identify running-best and configs
   already tried; check remaining budget (jobs/walltime).
2. **Hypothesize** — choose the next experiment, grounded in the ledger. Every ~5
   experiments, or after a 3-run no-improvement streak, delegate a fresh literature
   crawl to the `researcher` subagent before proposing the next config.
3. **Submit** — spawn `ml-worker` (submit-mode) with the hypothesis + config. It
   preflights and submits ONE job `--no-wait` and returns the `job_id` + flavor +
   timeout. Append a `submitted` ledger row. **Serial: one job in flight at a time.**
4. **Wait** — `ScheduleWakeup(delaySeconds ≈ expected job duration)`. On wake,
   `uv run ${CLAUDE_PLUGIN_ROOT}/scripts/hf_jobs.py inspect --job-id <job_id>`: if still
   running, reschedule; if done, continue.
5. **Evaluate** — spawn `ml-worker` (evaluate-mode) with the `job_id` + current
   running-best. It returns metric, Hub URL, and a keep/discard recommendation.
6. **Keep/discard** — update the ledger and `summary.md`; keep the best pushed
   model. Apply `program.md`'s keep/discard criteria.
7. **Stop or continue** — see below.

## Stop conditions

Stop the loop when ANY holds:
- the target metric is reached;
- the budget is exhausted (the hook will deny further jobs — treat a denial as a
  hard stop);
- a no-improvement streak exceeds the `program.md` limit;
- an unrecoverable error occurs.

Reserve the final ~10% of the budget for a clean final evaluation and a short
writeup in `summary.md` (best config, its metric, Hub URL) rather than launching
new experiments.

## Escalate to the user (the only human-in-the-loop)

You run unattended, but stop and report to the user on: a budget-hook **denial**
(envelope breach), an unrecoverable error, or a request that would exceed scope
(e.g. a dataset not in `program.md`). Never widen scope or budget on your own.

## On errors

Follow ml-research-task error recovery via the worker (OOM → smaller batch + more grad-accum
/ checkpointing / bigger flavor, without changing method/seqlen/model/dataset).
Never retry the identical failing thing. A hook denial is not an error to retry —
it is a stop signal.
