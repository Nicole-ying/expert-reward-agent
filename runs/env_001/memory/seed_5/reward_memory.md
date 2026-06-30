# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | distance_anchor + progress_reward + soft_landing_bonus + stability_penalty | -111.47 | 69.10 | -0.170 | progress 0.161; stability -0.245; soft 0.53% | failure | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -111.33 | 69.10 | -0.045 | progress 0.161; stability -0.126 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -111.33 | 69.10 | -0.045 | progress 0.161; stability -0.126 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -110.82 | 69.10 | 0.015 | progress 0.161; stability -0.065 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -110.82 | 69.10 | 0.015 | progress 0.161; stability -0.065 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 6 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | 3.10 | 993.10 | 3.524 | progress 0.029; stability -0.045 | failure | needs_review | inspect component balance |
| 7 | approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | -10.99 | 838.70 | 2.415 | progress 0.028; stability -0.024 | failure | early_failure_or_crash | add smoother approach/landing guidance |

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.

## Latest Iter Detail

### iter_7

- reward_structure: approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty
- external_score: -10.99
- mean_episode_length: 838.70
- reward_error_count: 0

#### component_evidence

- progress 0.028; stability -0.024

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance
