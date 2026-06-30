# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -107.82 | 72.00 | -0.183 | progress 0.160; stability -0.343 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 2 | distance_reward + landing_bonus + progress_reward + stability_penalty | -101.26 | 72.20 | -0.047 | progress 0.161; stability -0.117 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | approach_shaping + distance_reward + landing_bonus + progress_reward + stability_penalty | -42.47 | 1000.00 | 1.490 | progress 0.019; stability -0.022 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_reward + progress_reward + smooth_landing_shaping + stability_penalty | -5.48 | 1000.00 | 1.021 | progress 0.018; stability -0.011 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | contact_landing_bonus + distance_reward + progress_reward + smooth_landing_shaping + stability_penalty | 85.68 | 864.00 | 2.475 | progress 0.027; stability -0.005 | failure | needs_review | inspect component balance |
| 6 | contact_landing_bonus + distance_reward + progress_reward + smooth_landing_shaping + stability_penalty | 41.11 | 873.30 | 1.724 | progress 0.020; stability -0.002 | failure | needs_review | inspect component balance |
| 7 | contact_landing_bonus + distance_reward + progress_reward + smooth_landing_shaping + stability_penalty | 131.09 | 875.60 | 1.438 | progress 0.022; stability -0.001 | failure | needs_review | inspect component balance |
| 8 | contact_landing_bonus + distance_reward + progress_reward + smooth_landing_shaping | 125.44 | 937.90 | 1.205 | progress 0.023 | failure | needs_review | inspect component balance |
| 9 | contact_landing_bonus + distance_reward + progress_reward + smooth_landing_shaping | 125.44 | 937.90 | 1.205 | progress 0.023 | failure | needs_review | inspect component balance |
| 10 | contact_landing_bonus + progress_reward + smooth_landing_shaping | -8.24 | 1000.00 | 1.289 | progress 0.019 | failure | early_failure_or_crash | add smoother approach/landing guidance |

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

- reward_structure: contact_landing_bonus + progress_reward + smooth_landing_shaping
- external_score: -8.24
- mean_episode_length: 1000.00
- reward_error_count: 0

#### component_evidence

- progress 0.019

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance
