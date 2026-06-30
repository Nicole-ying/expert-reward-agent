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
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (keep, unchanged)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Reduced stability penalty (weaken coefficients to avoid dominating)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.2 * abs(next_body_angle)       # reduced from 0.5
    speed_penalty = 0.15 * speed                      # reduced from 0.3
    ang_vel_penalty = 0.1 * abs(next_ang_vel)         # reduced from 0.2
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Smooth landing-quality shaping (revised from hard sparse bonus)
    # Reward being near target with low speed and stable angle, continuous scale
    near_target = max(0.0, 1.0 - next_dist / 2.0)     # 1.0 when dist=0, 0.0 when dist>=2
    low_speed = max(0.0, 1.0 - speed / 1.0)           # 1.0 when speed=0, 0.0 when speed>=1
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # 1.0 when angle=0, 0.0 when angle>=0.5
    landing_quality = near_target * low_speed * stable_angle
    landing_shaping = 3.0 * landing_quality
    
    # 4. Small distance anchor to prevent drifting far away (keep)
    distance_anchor = -0.1 * next_dist
    
    # Combine components
    total_reward = progress_reward + stability_penalty + landing_shaping + distance_anchor
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Iteration Control Decision

- iteration_to_generate: 3
- target_score: 200.000
- best_score_so_far: -36.727
- best_iter: 2
- previous_score: -36.727
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
- mean_eval_reward: -36.727004
- mean_episode_length: 1000.000000
- min_eval_reward: -57.689342
- max_eval_reward: -3.071929

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.019753, nonzero_rate: 0.998053
- stability_penalty mean: -0.058407, abs_mean: 0.058407

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

### iter_2

- reward_structure: distance_anchor + landing_shaping + progress_reward + stability_penalty
- external_score: -36.73
- best_score_so_far: -36.73
- best_iter: 2
- mean_episode_length: 1000.00
- reward_error_count: 0
- verdict: failure
- decision: new_best

#### component_evidence

- progress 0.020; stability -0.058; distance -0.050

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
