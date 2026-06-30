# environment_contract

- env_id: Env_001
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed observation signals:
  - obs[0], next_obs[0]: x_position relative to target
  - obs[1], next_obs[1]: y_position relative to target
  - obs[2], next_obs[2]: x_velocity
  - obs[3], next_obs[3]: y_velocity
  - obs[4], next_obs[4]: body_angle
  - obs[5], next_obs[5]: angular_velocity
  - obs[6], next_obs[6]: left_support_contact
  - obs[7], next_obs[7]: right_support_contact
- action: discrete engine command, usable only as current action
- info: no reliable fields available
- forbidden: original_reward, official_reward, fitness_score, individual_reward, info['success'], info['failure'], info['termination_reason']
- terminal_success_reward and terminal_failure_penalty remain blocked unless explicit signals are added later


# previous_reward.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    old_x = obs[0]
    old_y = obs[1]
    new_x = next_obs[0]
    new_y = next_obs[1]
    new_vx = next_obs[2]
    new_vy = next_obs[3]
    new_angle = next_obs[4]
    new_angular_velocity = next_obs[5]

    old_distance = (old_x * old_x + old_y * old_y) ** 0.5
    new_distance = (new_x * new_x + new_y * new_y) ** 0.5
    next_speed = (new_vx * new_vx + new_vy * new_vy) ** 0.5

    progress_reward = old_distance - new_distance
    stability_penalty = -0.03 * next_speed - 0.03 * abs(new_angle) - 0.01 * abs(new_angular_velocity)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    if isinstance(info, dict):
        info["reward_terms"] = components

    return float(total_reward), components
```

# iteration_context.md

# Iteration Control Decision

- iteration_to_generate: 2
- target_score: 200.000
- best_score_so_far: -282.784
- best_iter: 1
- previous_score: -282.784
- no_improvement_count: 0
- trend: at_best
- decision: MUST_MODIFY
- required_action: tune/delete/add/mix based on failure evidence; do not return identical reward
- forbidden_action: no-op revision; generic restatement; changing comments only
- note: This control block has higher priority than matched expert cards.

# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: -282.784023
- mean_episode_length: 73.800000
- min_eval_reward: -440.645597
- max_eval_reward: -115.887255

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.011262, nonzero_rate: 1.000000
- stability_penalty mean: -0.037345, abs_mean: 0.037345

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length
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

### iter_1

- reward_structure: progress_reward + stability_penalty
- external_score: -282.78
- best_score_so_far: -282.78
- best_iter: 1
- mean_episode_length: 73.80
- reward_error_count: 0
- verdict: failure
- decision: new_best

#### component_evidence

- progress 0.011; stability -0.037

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance

## Matched Expert Cards

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
