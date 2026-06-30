# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | landing_bonus + progress_reward + stability_penalty | -111.60 | 71.40 | -0.180 | progress 0.032; stability -0.218 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 2 | distance_reward + landing_bonus + progress_reward + stability_penalty | -115.81 | 71.20 | -0.168 | progress 0.032; stability -0.110 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_reward + landing_bonus + progress_reward + stability_penalty | -114.10 | 71.30 | -0.114 | progress 0.032; stability -0.056 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_reward + landing_bonus + progress_reward + stability_penalty | -116.10 | 71.50 | -0.091 | progress 0.032; stability -0.034 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | distance_reward + landing_bonus + progress_reward + stability_penalty | -81.12 | 72.50 | -0.078 | progress 0.032; stability -0.022 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 6 | approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty | 177.00 | 789.70 | 0.467 | progress 0.006; stability -0.003 | failure | needs_review | inspect component balance |
| 7 | approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty | 174.21 | 886.80 | 0.494 | progress 0.006; stability -0.001 | failure | needs_review | inspect component balance |
| 8 | approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty | 107.08 | 1000.00 | 0.469 | progress 0.006; stability -0.001 | failure | needs_review | inspect component balance |
| 9 | approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty | 171.36 | 777.00 | 0.482 | progress 0.006; stability -0.000 | failure | needs_review | inspect component balance |
| 10 | approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty | 164.15 | 907.90 | 0.477 | progress 0.009; stability -0.000 | failure | needs_review | inspect component balance |

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.

## Latest Iter Detail

### iter_10

- reward_structure: approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty
- external_score: 164.15
- mean_episode_length: 907.90
- reward_error_count: 0

#### component_evidence

- progress 0.009; stability -0.000

#### diagnosis

- needs_review

#### next_action

- inspect component balance
