# `ml-research` plugin

The `ml-research` plugin is a reimplementation of Hugging Face's
[ml-intern](https://github.com/huggingface/ml-intern) as a native Claude Code
plugin. Same ml-intern behavior, but no custom harness: Claude Code is the
agent, and this plugin supplies the skills, subagents, and tools.

Hugging Face's ml-intern is an autonomous ML engineer: give it a goal and it
mines the literature for a recipe, validates the data, writes the training code,
and runs it on Hugging Face's cloud GPUs. It is a complete, self-contained
application, with its own agent loop, planning, file/shell tools, web search,
approvals, and web UI, and it calls an LLM per token through an API.

Claude Code already provides all of that scaffolding. So this rewrite keeps only
what is genuinely ml-intern's, recast as Claude Code skills, subagents, and
scripts:

- the research-first **playbook** → a skill (`ml-research-task`),
- the literature **research agent** → a subagent (`researcher`),
- the **Hugging Face / GitHub tool wrappers** → scripts Claude calls.

Everything else (the agent loop, planning, file/shell tools, web search,
approvals, the web UI) is just Claude Code. So every model call runs on your
Claude Code subscription and the tools talk only to Hugging Face and GitHub:
same behavior, no LLM API key.

Beyond ml-intern's per-task behavior, the rewrite adds its own **autonomous,
budget-bounded sweep** (`ml-research-loop`): many experiments run unattended
within limits you set.

## Install

Add the marketplace and install the plugin:

```bash
claude plugin marketplace add krasserm/ml-plugins
claude plugin install ml-research@ml-plugins
```

### Prerequisites

The plugin drives external tools, so it needs these available and authenticated:

- [`uv`](https://docs.astral.sh/uv/) runs the self-contained helper scripts.
- [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) logged in (or
  `HF_TOKEN` in a `.env`), used for Hugging Face access from the scripts and
  passed securely to your HF Jobs; cloud training needs **Jobs credits**.
- [`gh`](https://cli.github.com/) logged in, only for the GitHub code search.
- Optionally, set `S2_API_KEY` to avoid Semantic Scholar rate limits during
  literature research.

## Interactive use

Describe a task and the `ml-research-task` skill loads automatically and drives it:

> *"Fine-tune Qwen/Qwen2-0.5B on trl-lib/Capybara with LoRA on HF Jobs."
> "Find the best SFT recipe for a small instruct model from the literature."
> "Inspect krasserm/deepjob-clean-sft-v2."*

It works research-first: find the relevant papers and the recipes behind good
published results, validate the dataset and current library APIs, write the
script, preflight, submit **one** job, monitor it, and report the pushed model's
Hub URL. It pauses to confirm before anything that spends money or writes to a
repo.

## Autonomous sweeps

A *sweep* is a hands-off optimization run: the plugin repeatedly proposes an
experiment, trains it on HF Jobs, reads the metric, keeps the model only if it
beats the best so far, and uses what it learned to pick the next configuration.
It runs on its own (submitting a job, sleeping while it trains, then waking to
evaluate and start the next) until it hits your target metric or exhausts the
budget, then reports the best model. To start one, just ask:

> *"Run an autonomous sweep to minimize eval loss fine-tuning Qwen/Qwen2-0.5B on
> trl-lib/Capybara; cap it at 6 jobs on a10g-large and 4h total."*

A sweep is governed by a `program.md` in your project: the goal and target
metric, what it may touch (models, datasets, output repos), and a **budget
envelope** (max jobs, wall-clock, hardware, per-job timeout). If you don't have
one, the plugin **drafts it with you** and shows it for approval before spending
anything. Progress lands in `runs/summary.md`, and the run resumes cleanly if
interrupted.

## Safety & approvals

- **Interactive (the default):** the assistant pauses and asks for your
  confirmation before anything that spends money or changes a remote repo, namely
  submitting a cloud training job or uploading to / deleting from a Hub repo,
  stating the command, hardware, timeout, and a rough cost. You can pre-approve a
  batch upfront ("go ahead, run the jobs"); it then announces each action's cost
  before running instead of asking again.
- **Autonomous (a `program.md` is present):** you approve a budget up front and
  it is enforced on every job. Jobs within budget run uninterrupted; anything
  that would exceed it (too many jobs, a disallowed GPU, too long a timeout, an
  out-of-scope repo, or a delete you didn't allow) is blocked and reported back,
  never silently widened.

## Credits

Reimplements Hugging Face's
[`ml-intern`](https://github.com/huggingface/ml-intern) (Apache-2.0), which the
tooling, skill, and `researcher` subagent adapt. Also draws on
[`universal-ml-intern`](https://github.com/mudler/universal-ml-intern) for the
harness-native approach and the `program.md` convention.
