# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: -111.544763
- mean_episode_length: 74.100000
- min_eval_reward: -121.825106
- max_eval_reward: -105.813061

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.159844, nonzero_rate: 0.999994
- speed_penalty mean: -0.525343, abs_mean: 0.525343, nonzero_rate: 0.999928
- stability_penalty mean: -0.024591, abs_mean: 0.024591
- soft_landing_bonus mean: 0.008542, trigger_rate: 0.004271
- total_reward mean: -0.381548, abs_mean: 0.397792
- generated_reward mean: -0.381548; original_env_reward mean: -1.543968

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length
- likely_issue: speed_penalty dominates progress signal
- evidence: abs(speed_penalty mean)=0.525343 > abs(progress_reward mean)=0.159844
- likely_issue: soft_landing_bonus is too sparse or rarely reached
- evidence: trigger_rate=0.004271
- likely_issue: generated reward is negative on average during training

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

## Matched Expert Cards

## speed_penalty_dominance

- signal: abs(speed_penalty_mean) > abs(progress_reward_mean)
- risk: agent may learn to avoid motion instead of solving the task
- fix: reduce speed penalty, delay it, or apply it mainly near landing-critical states

## sparse_completion_proxy

- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

## early_failure_or_crash

- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

## generated_reward_negative_average

- signal: total generated reward mean is negative during most training steps
- risk: agent receives mostly punitive feedback and may not discover useful behavior
- fix: rebalance progress and penalty terms; avoid large always-on penalties in early versions

## Skeleton Revision Plan

### keep
- progress_delta_reward

### weaken
- speed_penalty

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
