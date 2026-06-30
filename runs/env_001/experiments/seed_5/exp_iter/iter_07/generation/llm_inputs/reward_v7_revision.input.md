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
    
    # 1. 主学习信号：progress_delta_reward (保留)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定约束：stability_penalty (保留)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.2 * abs(next_body_angle)
    angular_vel_penalty = -0.1 * abs(next_angular_vel)
    speed_penalty = -0.05 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_shaping_reward (修订为加性组合)
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    # 加性组合：每个良好行为独立贡献奖励
    landing_shaping_reward = 2.0 * near_target + 1.0 * low_speed + 1.0 * stable_angle + 1.0 * both_contact
    
    # 4. 接近奖励：approach_reward (新增)
    # 奖励向目标移动，不依赖着陆状态
    approach_reward = 2.0 * max(0.0, 1.0 - next_dist / 1.0)
    
    # 5. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + approach_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "approach_reward": approach_reward,
        "distance_anchor": distance_anchor,
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
- mean_eval_reward: 3.095448
- mean_episode_length: 993.100000
- min_eval_reward: -48.823973
- max_eval_reward: 118.003229

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.028581, nonzero_rate: 0.999748
- stability_penalty mean: -0.044895, abs_mean: 0.044895

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

### iter_6

- reward_structure: approach_reward + distance_anchor + landing_shaping_reward + progress_reward + stability_penalty
- external_score: 3.10
- mean_episode_length: 993.10
- reward_error_count: 0

#### component_evidence

- progress 0.029; stability -0.045

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
