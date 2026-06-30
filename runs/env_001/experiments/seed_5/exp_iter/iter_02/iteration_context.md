# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: -111.465343
- mean_episode_length: 69.100000
- min_eval_reward: -121.749325
- max_eval_reward: -96.245274

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.161238, nonzero_rate: 0.999994
- stability_penalty mean: -0.244504, abs_mean: 0.244504
- soft_landing_bonus mean: 0.010670, trigger_rate: 0.005335

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length
- likely_issue: soft_landing_bonus is too sparse or never reached
- likely_issue: stability penalty may dominate progress signal

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

### iter_1

- reward_structure: distance_anchor + progress_reward + soft_landing_bonus + stability_penalty
- external_score: -111.47
- mean_episode_length: 69.10
- reward_error_count: 0

#### component_evidence

- progress 0.161; stability -0.245; soft 0.53%

#### diagnosis

- early_failure_or_crash; sparse_completion_proxy

#### next_action

- add smoother approach/landing guidance; smooth landing proxy

## Matched Expert Cards

## sparse_completion_proxy

- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

## early_failure_or_crash

- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

## Skeleton Revision Plan

### keep
- progress_delta_reward

### weaken
- none

### revise
- soft_landing_proxy -> smoother landing-quality shaping

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
