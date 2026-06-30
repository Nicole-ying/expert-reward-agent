# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | best_so_far | delta_from_best | len | gen_reward | key_component_signal | verdict | decision | diagnosis | next_action |
|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 1 | progress_reward + stability_penalty | -282.78 | -282.78 | 0.00 | 73.80 | -0.026 | progress 0.011; stability -0.037 | failure | new_best | early_failure_or_crash | add smoother approach/landing guidance |
| 2 | distance_anchor + landing_quality + progress_reward + speed_penalty + stability_penalty | -281.91 | -282.78 | 0.87 | 74.50 | -0.069 | progress 0.068; speed -0.075; stability -0.031; distance -0.051; landing 0.019 | failure | no_meaningful_improvement | early_failure_or_crash | add smoother approach/landing guidance |

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Target solved threshold for Env_001: mean external evaluation score >= 200.
- Preserve best-so-far reward; final reward should be the best reward, not necessarily the last reward.
- If the task has been solved and a later revision drops below target, stop and keep the best reward.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, score, best-so-far, decision, diagnosis, and next action only.

## Latest Iter Detail

### iter_2

- reward_structure: distance_anchor + landing_quality + progress_reward + speed_penalty + stability_penalty
- external_score: -281.91
- best_score_so_far: -282.78
- best_iter: 1
- mean_episode_length: 74.50
- reward_error_count: 0
- verdict: failure
- decision: no_meaningful_improvement

#### component_evidence

- progress 0.068; speed -0.075; stability -0.031; distance -0.051; landing 0.019

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance
