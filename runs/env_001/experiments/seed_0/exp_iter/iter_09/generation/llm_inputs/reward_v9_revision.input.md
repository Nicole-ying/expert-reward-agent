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
    # Extract relevant observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # Component 1: Progress delta reward (main learning signal)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Kept: progress_reward remains the primary driver.
    progress_reward = progress_delta * 8.0
    
    # Component 2: Proximity reward (smooth approach guidance)
    # Kept: proximity_reward provides a persistent, smooth gradient.
    proximity_reward = 1.0 / (1.0 + 3.0 * next_dist)
    
    # Component 3: Stability penalty (weakened to avoid dominating progress)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.05
    angular_vel_penalty = abs(next_angular_vel) * 0.025
    speed_penalty = speed * 0.025
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # Component 4: Conditional speed reward (encourages movement when near target)
    # Kept: conditional_speed_reward helps prevent "freezing" at the goal.
    conditional_speed_reward = 0.0
    if next_dist < 0.5:
        target_speed = 0.5
        speed_diff = abs(speed - target_speed)
        conditional_speed_reward = max(0.0, 0.2 - speed_diff * 0.4)
    
    # Component 5: Small action penalty (efficiency)
    # Kept: action_penalty provides a minor efficiency signal.
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # Combine components
    total_reward = progress_reward + proximity_reward + stability_penalty + conditional_speed_reward + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "proximity_reward": proximity_reward,
        "stability_penalty": stability_penalty,
        "conditional_speed_reward": conditional_speed_reward,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: 186.897804
- mean_episode_length: 772.400000
- min_eval_reward: 122.502855
- max_eval_reward: 283.914902

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.017074, nonzero_rate: 0.999983
- stability_penalty mean: -0.012772, abs_mean: 0.012772

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

### iter_8

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
