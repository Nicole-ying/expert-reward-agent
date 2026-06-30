# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | best_so_far | delta_from_best | len | gen_reward | key_component_signal | verdict | decision | diagnosis | next_action |
|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 1 | distance_anchor + progress_reward + soft_landing_bonus + stability_penalty | -111.47 | 69.10 | -0.170 | progress 0.161; stability -0.245; soft 0.53% | failure | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -111.33 | 69.10 | -0.045 | progress 0.161; stability -0.126 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -111.33 | 69.10 | -0.045 | progress 0.161; stability -0.126 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -110.82 | 69.10 | 0.015 | progress 0.161; stability -0.065 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -110.82 | 69.10 | 0.015 | progress 0.161; stability -0.065 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 6 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | 3.10 | 993.10 | 3.524 | progress 0.029; stability -0.045 | failure | needs_review | inspect component balance |
| 7 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -10.99 | 838.70 | 2.415 | progress 0.028; stability -0.024 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 8 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | 115.44 | 812.40 | 1.792 | progress 0.032; stability -0.006 | failure | needs_review | inspect component balance |
| 9 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | 219.68 | 511.20 | 1.823 | progress 0.031; stability -0.003 | success | needs_review | inspect component balance |
| 10 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -22.52 | -22.52 | 0.00 | 1000.00 | 1.251 | progress 0.044; stability -0.003; distance -0.037 | failure | continue | early_failure_or_crash | add smoother approach/landing guidance |

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

### iter_10

- reward_structure: approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty
- external_score: -22.52
- best_score_so_far: -22.52
- best_iter: 10
- mean_episode_length: 1000.00
- reward_error_count: 0
- verdict: failure
- decision: continue

#### component_evidence

- progress 0.044; stability -0.003; distance -0.037

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance
