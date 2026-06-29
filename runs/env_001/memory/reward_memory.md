# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -110.57 | 74.10 | -0.172 | progress 0.161; stability -0.342; soft 0.44% | failure | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | landing_shaping_reward + progress_reward + stability_penalty | -155.31 | 1000.00 | 0.723 | progress 0.008; stability -0.052 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | distance_anchor + landing_shaping_reward + progress_reward + stability_penalty | 55.59 | 371.10 | 0.557 | progress 0.063; stability -0.056 | failure | needs_review | inspect component balance |

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.

## Latest Iter Detail

### iter_3

- reward_structure: distance_anchor + landing_shaping_reward + progress_reward + stability_penalty
- external_score: 55.59
- mean_episode_length: 371.10
- reward_error_count: 0

#### component_evidence

- progress 0.063; stability -0.056

#### diagnosis

- needs_review

#### next_action

- inspect component balance
