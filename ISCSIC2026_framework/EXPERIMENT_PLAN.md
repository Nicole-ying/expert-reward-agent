# Experiment rerun and ablation audit

The existing package is sufficient to reproduce the paper-v4 main condition
and the current ablations. Before replacing numbers in the manuscript, rerun
every compared search method with the same five search lineages, maximum ten
completed reward evaluations per lineage, PPO budget, evaluation episodes, and
stopping threshold.

## Required comparison set

| Condition | Question answered | Entry point |
|---|---|---|
| CREATE | Full closed-loop agent | `scripts/run_paper_v4.sh` |
| Independent generation | Prompt- and budget-matched search without repair | `scripts/run_independent_baseline.sh` |
| Score-only feedback | Is the native score alone enough to diagnose repair? | `scripts/run_ablation_score_only_v4.sh` |
| Coarse feedback | Do component means substitute for structured evidence? | `scripts/run_ablation_eureka_feedback_v4.sh` |
| Unconstrained editing | Does removing the bounded edit contract destabilize repair? | `scripts/run_ablation_unconstrained_v4.sh` |
| Native-reward PPO | Environment-reward reference, not an LLM search baseline | `scripts/run_official_baseline.sh` |

## Important interpretation rule

The current `unconstrained` condition removes the complete reflection contract,
so it jointly changes target scope and edit hierarchy. It is valid as a system
ablation but does **not** isolate the causal contribution of single-target
editing from L1/L2/L3 hierarchy. If compute permits, add two follow-up
conditions before making a component-level causal claim:

1. structured evidence + single target + no L1/L2/L3 declaration;
2. structured evidence + L1/L2/L3 declaration + multiple allowed targets.

Until those runs exist, the paper should claim that structured evidence and
bounded revision are jointly important, rather than assigning the gain to one
subcomponent.

## Baseline requirement

The native-reward PPO reference measures the task's standard reward, but it is
not a fair replacement for a reward-search baseline. The included search
baseline is prompt- and budget-matched independent reward generation: the same
task context and LLM, a fresh reward at every evaluation, no lineage memory,
and no training-evidence-conditioned repair. Record successful lineages,
trainings-to-threshold, selected reward scores, and the final fixed validation
protocol for both methods.

## Outputs to retain

- per-iteration reward source and validation report;
- PPO configuration and random seed;
- native evaluation mean and dispersion;
- structured component statistics and training trace;
- diagnosis, declared edit level, primary target, and memory record;
- best archive and the evaluation at which it first crossed threshold;
- TensorBoard event files under `runs/env_001/tensorboard/`.

TensorBoard event files are machine-readable and can be converted into aligned
CSV tables and publication figures; retain them together with each run's JSON
summaries rather than copying values manually.
