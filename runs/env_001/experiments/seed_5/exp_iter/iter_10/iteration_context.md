# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: 219.676378
- mean_episode_length: 511.200000
- min_eval_reward: 163.808548
- max_eval_reward: 251.849040

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.031382, nonzero_rate: 0.999866
- stability_penalty mean: -0.003188, abs_mean: 0.003188

## 5. Preliminary failure hints

## Short Memory

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.

## Latest Iter Detail

### iter_9

- reward_structure: approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty
- external_score: 219.68
- mean_episode_length: 511.20
- reward_error_count: 0

#### component_evidence

- progress 0.031; stability -0.003

#### diagnosis

- needs_review

#### next_action

- inspect component balance

## Matched Expert Cards

- none

## Skeleton Revision Plan

### keep
- progress_delta_reward

### weaken
- none

### revise
- none

### consider_add
- distance_reward as a small anchor if progress-only guidance remains weak

### still_defer
- terminal_success_reward
- terminal_failure_penalty
- energy_penalty
- time_penalty
- gated_reward

## Reward Revision Boundary

- Revise the previous reward instead of generating from scratch.
- Keep the function signature unchanged.
- Do not use original_reward or unavailable info fields.
- Do not add terminal success/failure rewards without explicit signals.
- Prefer fewer components with clear roles over adding many skeletons.
