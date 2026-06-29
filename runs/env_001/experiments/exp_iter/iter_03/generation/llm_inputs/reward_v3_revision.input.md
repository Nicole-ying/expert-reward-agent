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
    
    # 1. Main learning signal: progress_delta_reward (kept)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Smoother landing-quality shaping (revised from soft_landing_bonus)
    # Use a continuous bonus that increases as the agent gets closer to the target,
    # moves slower, and maintains a more upright posture.
    # This replaces the hard sparse bonus with a smoother gradient.
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    landing_quality = 0.0
    # Distance component: reward being close to target
    distance_factor = max(0.0, 1.0 - next_dist / 0.5)  # 0 when far, 1 when at target
    landing_quality += 1.0 * distance_factor
    # Speed component: reward low speed
    speed_factor = max(0.0, 1.0 - speed / 0.5)  # 0 when fast, 1 when still
    landing_quality += 1.0 * speed_factor
    # Angle component: reward upright posture
    angle_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.2)  # 0 when tilted, 1 when upright
    landing_quality += 1.0 * angle_factor
    # Contact component: small bonus for having both feet on the ground
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_bonus = 0.5 if both_contact else 0.0
    landing_quality += contact_bonus
    
    # Scale the landing quality to provide meaningful guidance without dominating
    landing_shaping_reward = 0.5 * landing_quality
    
    # 3. Stability penalty (kept but reduced weight to prevent dominance)
    angle_penalty = -0.3 * abs(next_body_angle)
    angular_vel_penalty = -0.1 * abs(next_angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # Total reward
    total_reward = progress_reward + stability_penalty + landing_shaping_reward
    
    # Components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
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
- mean_eval_reward: -155.310526
- mean_episode_length: 1000.000000
- min_eval_reward: -178.536062
- max_eval_reward: -112.610211

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.007651, nonzero_rate: 0.999960
- stability_penalty mean: -0.052379, abs_mean: 0.052379

## 5. Preliminary failure hints
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

### iter_2

- reward_structure: landing_shaping_reward + progress_reward + stability_penalty
- external_score: -155.31
- mean_episode_length: 1000.00
- reward_error_count: 0

#### component_evidence

- progress 0.008; stability -0.052

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
