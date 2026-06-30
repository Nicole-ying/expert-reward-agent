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
    # 提取观测变量
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
    
    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty (进一步削弱权重)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)      # 从0.02降至0.01
    angular_vel_penalty = 0.005 * abs(next_ang_vel)   # 从0.01降至0.005
    speed_penalty = 0.005 * speed                     # 从0.01降至0.005
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (权重降低，阈值放宽)
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)  # 阈值从1.5改回2.0，更早开始
    speed_comfort = max(0.0, 1.0 - speed / 0.8)           # 保持不变
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)  # 保持不变
    smooth_landing_shaping = 2.0 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)  # 权重从3.0降至2.0
    
    # 4. 接触着陆奖励：contact_landing_bonus (简化，提高触发率)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 1.0  # 阈值从0.5放宽到1.0
    if contact and near_target:
        contact_bonus = 1.0  # 固定小奖励，避免乘积几乎为零
    else:
        contact_bonus = 0.0
    
    # 5. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + smooth_landing_shaping + contact_bonus + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "smooth_landing_shaping": smooth_landing_shaping,
        "contact_landing_bonus": contact_bonus,
        "distance_reward": distance_reward,
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
- mean_eval_reward: 41.110474
- mean_episode_length: 873.300000
- min_eval_reward: -112.144512
- max_eval_reward: 141.234219

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.019945, nonzero_rate: 0.999157
- stability_penalty mean: -0.002220, abs_mean: 0.002220

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

### iter_6

- reward_structure: contact_landing_bonus + distance_reward + progress_reward + smooth_landing_shaping + stability_penalty
- external_score: 41.11
- mean_episode_length: 873.30
- reward_error_count: 0

#### component_evidence

- progress 0.020; stability -0.002

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
