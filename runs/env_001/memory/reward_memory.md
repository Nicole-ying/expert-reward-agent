# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.

## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
| 1 | progress + stability + soft_landing | -110.84 | 74.1 | N/A | component stats missing; soft proxy likely sparse | failure | early_failure_or_crash | add component logging |
| 2 | progress + speed_penalty + stability + soft_landing | -111.54 | 74.1 | -0.382 | speed -0.525 > progress +0.160; soft trigger 0.43% | failure | speed_penalty_dominance; sparse_completion_proxy | weaken speed penalty; smooth landing proxy |

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.

## Latest Iter Detail

### iter_2

- reward_structure: progress + speed_penalty + stability + soft_landing
- external_score: -111.54
- mean_episode_length: 74.1
- reward_error_count: 0

#### component_evidence

- progress_reward mean: +0.159844
- speed_penalty mean: -0.525343
- stability_penalty mean: -0.024591
- soft_landing_trigger_rate: 0.004271
- total_generated_reward mean: -0.381548

#### skeleton_decision

- keep: progress_delta_reward
- weaken: speed_penalty
- revise: soft_landing_proxy toward smoother landing-quality shaping
- consider_add: small distance anchor if progress-only guidance remains weak
- still_defer: terminal_success_reward, terminal_failure_penalty, energy_penalty, time_penalty, gated_reward
