# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | progress_delta_reward + soft_landing_bonus + stability_penalty | -109.62 | 70.60 | -0.170 | stability -0.342; soft 0.54% | failure | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | distance_anchor_reward + landing_quality_shaping + progress_delta_reward | -110.62 | 70.60 | -0.021 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_anchor_reward + landing_quality_shaping + progress_delta_reward | -110.62 | 70.60 | -0.021 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_anchor_reward + landing_quality_shaping + progress_delta_reward | -105.06 | 70.70 | 0.021 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | distance_anchor_reward + progress_delta_reward + smooth_landing_bonus | 6.34 | 947.90 | 0.673 | component stats unavailable | failure | needs_review | inspect component balance |
| 6 | distance_anchor_reward + progress_delta_reward + smooth_landing_bonus | -38.66 | 1000.00 | 2.079 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 7 | distance_anchor_reward + progress_delta_reward + smooth_landing_bonus | 45.61 | 864.90 | 0.805 | component stats unavailable | failure | needs_review | inspect component balance |
| 8 | distance_anchor_reward + progress_delta_reward + smooth_landing_bonus | -9.00 | 1000.00 | 0.082 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 9 | distance_anchor_reward + progress_delta_reward + smooth_landing_bonus | -480.61 | 65.80 | -0.273 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 10 | distance_anchor_reward + progress_delta_reward + smooth_landing_bonus | -75.44 | 98.90 | -0.213 | component stats unavailable | failure | early_failure_or_crash | add smoother approach/landing guidance |

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

- reward_structure: distance_anchor_reward + progress_delta_reward + smooth_landing_bonus
- external_score: -75.44
- mean_episode_length: 98.90
- reward_error_count: 0

#### component_evidence

- component stats unavailable

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance
