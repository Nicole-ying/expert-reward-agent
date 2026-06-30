# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | best_so_far | delta_from_best | len | gen_reward | key_component_signal | verdict | decision | diagnosis | next_action |
|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 1 | distance_anchor + progress_reward + soft_landing_bonus + stability_penalty | -112.27 | -112.27 | 0.00 | 74.10 | -0.266 | progress 0.161; stability -0.342; distance -0.097; soft 0.54% | failure | new_best | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -36.73 | -36.73 | 0.00 | 1000.00 | 1.528 | progress 0.020; stability -0.058; distance -0.050 | failure | new_best | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -222.10 | -36.73 | -185.37 | 1000.00 | 2.199 | progress 0.028; stability -0.037; distance -0.054 | failure | no_meaningful_improvement | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -41.34 | -36.73 | -4.61 | 1000.00 | 4.263 | progress 0.067; stability -0.022; distance -0.114 | failure | no_meaningful_improvement | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | distance_anchor + landing_shaping + progress_reward + stability_penalty | 18.32 | 18.32 | 0.00 | 954.00 | 2.414 | progress 0.194; stability -0.005; distance -0.025 | failure | new_best | needs_review | inspect component balance and score trend |
| 6 | distance_reward + landing_shaping + progress_reward + stability_penalty | 146.23 | 146.23 | 0.00 | 840.70 | 2.718 | progress 0.548; stability -0.005 | failure | new_best | needs_review | inspect component balance and score trend |
| 7 | distance_reward + landing_shaping + progress_reward + stability_penalty | -15.09 | 146.23 | -161.32 | 1000.00 | 2.531 | progress 0.537; stability -0.005 | failure | no_meaningful_improvement | early_failure_or_crash | add smoother approach/landing guidance |
| 8 | distance_reward + landing_shaping + progress_reward + stability_penalty | -18.55 | 146.23 | -164.78 | 1000.00 | 4.571 | progress 0.777; stability -0.006 | failure | no_meaningful_improvement | early_failure_or_crash | add smoother approach/landing guidance |
| 9 | distance_reward + landing_shaping + progress_reward + stability_penalty | -69.88 | 146.23 | -216.11 | 1000.00 | 6.127 | progress 0.256; stability -0.006 | failure | unsolved_stagnation_fresh_restart | early_failure_or_crash | add smoother approach/landing guidance |

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

### iter_9

- reward_structure: distance_reward + landing_shaping + progress_reward + stability_penalty
- external_score: -69.88
- best_score_so_far: 146.23
- best_iter: 6
- mean_episode_length: 1000.00
- reward_error_count: 0
- verdict: failure
- decision: unsolved_stagnation_fresh_restart

#### component_evidence

- progress 0.256; stability -0.006

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance
