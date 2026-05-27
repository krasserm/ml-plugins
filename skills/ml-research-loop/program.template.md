# program.md — autonomous run task spec (template)

Copy this to `program.md` at the repo root to run `ml-research-loop` autonomously. It is
the externalized contract: the loop reads it every iteration (surviving context
compaction), and the budget hook enforces the envelope. With NO `program.md` present, the project behaves normally (interactive,
prompt-gated) — this file is what flips it into bounded autonomous mode.

Keep the **Budget envelope** block exactly as a fenced ```yaml block under the
`## Budget envelope` heading; that is the part the hook parses.

---

## Goal

One measurable outcome. State the metric and the target.
> e.g. "Maximize eval accuracy of an SFT of `google/gemma-4-E2B-it` on
> `krasserm/deepjob-clean-sft-v2`; target ≥ 0.85 on the validation split."

## Scope

What the loop may and may not touch.
- Models allowed: `<base model id(s)>`
- Datasets allowed: `<dataset id(s)>` — never silently substitute.
- Output repos (where trained models/adapters are pushed): `<owner/prefix-*>`
- Off-limits: anything not listed here.

## Reference artifacts

Known-good starting points the loop should build from.
- Reference implementation / example script: `<path or URL>`
- Baseline result (if any): `<metric = value>`

## Output format

Where results live so the run is resumable and auditable.
- Ledger: `runs/ledger.jsonl` (one row per experiment; appended by the loop)
- Summary: `runs/summary.md` (human-readable running table)
- Metric source: the grep-able plain-text line the training script logs
  (e.g. `eval_accuracy=`), per the ml-research-task logging convention.

## Keep/discard criteria

When is an experiment kept vs discarded, and when is the whole run done.
- Keep an experiment if: `<metric>` beats the running best.
- Discard if: it errors, diverges, or does not beat the running best.
- Run is DONE when: target metric reached, OR budget exhausted, OR
  `<N>` consecutive experiments with no improvement.

## Budget envelope

The machine-readable authorization. The budget hook auto-approves
`hf_jobs.py run` only while within these limits, and denies (escalates) on breach.

```yaml
max_jobs: 12                       # total HF Jobs the run may submit
max_walltime: 8h                   # total job wall-clock across the run
allowed_flavors:                   # hardware flavors the loop may request
  - t4-small
  - a10g-large
max_timeout: 2h                    # max --timeout per job
allow_deletes: false               # if false, remote deletes are blocked
output_repos:                      # repos the loop may write to (glob ok)
  - me/sweep-*
```
