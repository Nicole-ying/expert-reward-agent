# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | distance_anchor + progress_delta_reward + soft_landing_bonus + stability_penalty | -110.44 | 73.60 | -0.269 | stability -0.345; soft 0.65% | failure | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty | -39.02 | 1000.00 | 1.556 | stability -0.069 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty | -39.02 | 1000.00 | 1.556 | stability -0.069 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty | -21.82 | 1000.00 | 2.184 | stability -0.071 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 5 | distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty | 231.29 | 459.30 | 0.683 | stability -0.026 | success | needs_review | inspect component balance |
| 6 | distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty | 231.29 | 459.30 | 0.683 | stability -0.026 | success | needs_review | inspect component balance |
| 7 | distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty | 259.52 | 335.40 | 0.698 | stability -0.013 | success | needs_review | inspect component balance |
| 8 | distance_reward + landing_shaping_reward + progress_delta_reward | 249.42 | 374.50 | 1.014 | component stats unavailable | success | needs_review | inspect component balance |
| 9 | distance_reward + landing_shaping_reward + progress_delta_reward | 21.92 | 963.80 | 0.961 | component stats unavailable | failure | needs_review | inspect component balance |
| 10 | distance_reward + landing_shaping_reward + progress_delta_reward | 167.43 | 935.80 | 0.370 | component stats unavailable | failure | needs_review | inspect component balance |

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

- reward_structure: distance_reward + landing_shaping_reward + progress_delta_reward
- external_score: 167.43
- mean_episode_length: 935.80
- reward_error_count: 0

#### component_evidence

- component stats unavailable

#### diagnosis

- needs_review

#### next_action

- inspect component balance
