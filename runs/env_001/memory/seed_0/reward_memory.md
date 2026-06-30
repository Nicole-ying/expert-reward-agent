# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | action_penalty + progress_reward + soft_landing_bonus + stability_penalty | -112.85 | 74.10 | -0.075 | progress 0.161; stability -0.242; soft 0.65% | failure | early_failure_or_crash; sparse_completion_proxy | add smoother approach/landing guidance; smooth landing proxy |
| 2 | action_penalty + distance_anchor + landing_shaping + progress_reward + stability_penalty | -110.90 | 74.10 | 0.075 | progress 0.161; stability -0.116 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 3 | action_penalty + distance_anchor + landing_shaping + progress_reward + stability_penalty | -110.90 | 74.10 | 0.075 | progress 0.161; stability -0.116 | failure | early_failure_or_crash | add smoother approach/landing guidance |
| 4 | action_penalty + distance_anchor + progress_reward + proximity_reward + stability_penalty | 204.08 | 726.90 | 0.536 | progress 0.037; stability -0.023 | success | needs_review | inspect component balance |
| 5 | action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty | 249.71 | 381.30 | 0.592 | progress 0.024; stability -0.011 | success | needs_review | inspect component balance |
| 6 | action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty | 249.71 | 381.30 | 0.592 | progress 0.024; stability -0.011 | success | needs_review | inspect component balance |
| 7 | action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty | 186.90 | 772.40 | 0.670 | progress 0.017; stability -0.013 | failure | needs_review | inspect component balance |
| 8 | action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty | 186.90 | 772.40 | 0.670 | progress 0.017; stability -0.013 | failure | needs_review | inspect component balance |
| 9 | action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty | 186.90 | 772.40 | 0.670 | progress 0.017; stability -0.013 | failure | needs_review | inspect component balance |
| 10 | action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty | 186.90 | 772.40 | 0.670 | progress 0.017; stability -0.013 | failure | needs_review | inspect component balance |

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

- reward_structure: action_penalty + conditional_speed_reward + progress_reward + proximity_reward + stability_penalty
- external_score: 186.90
- mean_episode_length: 772.40
- reward_error_count: 0

#### component_evidence

- progress 0.017; stability -0.013

#### diagnosis

- needs_review

#### next_action

- inspect component balance
