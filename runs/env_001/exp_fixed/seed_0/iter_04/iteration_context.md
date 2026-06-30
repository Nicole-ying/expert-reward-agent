# Iteration Control Decision

- iteration_to_generate: 4
- target_score: 200.000
- best_score_so_far: -36.727
- best_iter: 2
- previous_score: -222.100
- no_improvement_count: 1
- trend: below_best
- decision: MUST_MODIFY
- required_action: tune/delete/add/mix based on failure evidence; do not return identical reward
- forbidden_action: no-op revision; generic restatement; changing comments only
- note: This control block has higher priority than matched expert cards.

# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: -222.099862
- mean_episode_length: 1000.000000
- min_eval_reward: -246.073444
- max_eval_reward: -196.583370

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.028210, nonzero_rate: 0.999919
- stability_penalty mean: -0.036711, abs_mean: 0.036711

## 5. Preliminary failure hints
- likely_issue: stability penalty may dominate progress signal

## Short Memory

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Target solved threshold for Env_001: mean external evaluation score >= 200.
- Preserve best-so-far reward; final reward should be the best reward, not necessarily the last reward.
- If the task has been solved and a later revision drops below target, stop and keep the best reward.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, score, best-so-far, decision, diagnosis, and next action only.

## Latest Iter Detail

### iter_3

- reward_structure: distance_anchor + landing_shaping + progress_reward + stability_penalty
- external_score: -222.10
- best_score_so_far: -36.73
- best_iter: 2
- mean_episode_length: 1000.00
- reward_error_count: 0
- verdict: failure
- decision: no_meaningful_improvement

#### component_evidence

- progress 0.028; stability -0.037; distance -0.054

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance

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
