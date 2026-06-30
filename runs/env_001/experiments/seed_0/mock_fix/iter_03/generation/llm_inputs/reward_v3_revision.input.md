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
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = 6.0 * (current_dist - next_dist)
    distance_anchor = -0.05 * next_dist

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty = -0.08 * speed
    stability_penalty = -0.08 * abs(next_obs[4]) - 0.04 * abs(next_obs[5])

    near_target_quality = 1.0 / (1.0 + 4.0 * next_dist)
    low_speed_quality = 1.0 / (1.0 + 2.0 * speed)
    angle_quality = 1.0 / (1.0 + 5.0 * abs(next_obs[4]))
    landing_quality = 0.4 * near_target_quality * low_speed_quality * angle_quality

    total_reward = progress_reward + distance_anchor + speed_penalty + stability_penalty + landing_quality
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "speed_penalty": speed_penalty,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```

# iteration_context.md

# Iteration Control Decision

- iteration_to_generate: 3
- target_score: 200.000
- best_score_so_far: -282.784
- best_iter: 1
- previous_score: -281.910
- no_improvement_count: 1
- trend: unsolved_search
- decision: MUST_MODIFY
- required_action: tune/delete/add/mix based on failure evidence; do not return identical reward
- forbidden_action: no-op revision; generic restatement; changing comments only
- note: This control block has higher priority than matched expert cards.

# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: -281.909661
- mean_episode_length: 74.500000
- min_eval_reward: -519.526345
- max_eval_reward: -158.204128

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.068404, nonzero_rate: 1.000000
- stability_penalty mean: -0.030860, abs_mean: 0.030860

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length

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

### iter_2

- reward_structure: distance_anchor + landing_quality + progress_reward + speed_penalty + stability_penalty
- external_score: -281.91
- best_score_so_far: -282.78
- best_iter: 1
- mean_episode_length: 74.50
- reward_error_count: 0
- verdict: failure
- decision: no_meaningful_improvement

#### component_evidence

- progress 0.068; speed -0.075; stability -0.031; distance -0.051; landing 0.019

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


## No-op Retry Instruction

The previous revision attempt #1 produced a reward function that is semantically identical to previous_reward.py.
This is not an acceptable iteration while the task is still unsolved.
You must perform a concrete tune/delete/add/mix action justified by the training evidence.
Do not return identical reward code again.


## No-op Retry Instruction

The previous revision attempt #2 produced a reward function that is semantically identical to previous_reward.py.
This is not an acceptable iteration while the task is still unsolved.
You must perform a concrete tune/delete/add/mix action justified by the training evidence.
Do not return identical reward code again.
