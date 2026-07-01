# 环境契约
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty


# 当前奖励函数代码
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
    
    # 1. Main learning signal: progress_delta_reward
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty - reduced by 20x and distance-gated
    # Only penalize instability when near the target (dist < 3.0)
    # Far from target, let the agent move freely
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.025 * abs(next_body_angle)
    angular_vel_penalty = -0.015 * abs(next_angular_vel)
    speed_penalty = -0.01 * speed
    
    # Distance gate: only apply stability penalty when near target
    gate = 1.0 / (1.0 + 2.0 * next_dist)  # ~1 when dist=0, ~0.2 when dist=2, ~0.09 when dist=5
    stability_penalty = gate * (angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Each factor is in [0,1], product gives smooth gradient
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)  # bounded [0,1], high when near target
    speed_factor = 1.0 / (1.0 + 5.0 * speed)  # bounded [0,1], high when slow
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))  # bounded [0,1], high when upright
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))  # bounded [0,1], low angular vel
    
    # Contact factor: both feet on ground
    contact_factor = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_factor = float(contact_factor)  # 0 or 1, but that's okay as a gate
    
    soft_landing_bonus = 5.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty for using engines (action != 0)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    # Combine rewards
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 历史最佳奖励函数代码
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
    
    # 1. Main learning signal: progress_delta_reward
    # Reward for moving closer to the target (origin in relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: encourage low velocity, upright angle, low angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)
    angular_vel_penalty = -0.3 * abs(next_angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: small bonus when near target, stable, and both supports contact
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small energy penalty for using engines (action != 0)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    # Combine rewards
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-118.437388, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.004933 | 0.004933 | 0.098654 | -0.030456 |
| progress_delta_reward | 0.161961 | 0.171098 | 0.999998 | 1.000000 |
| soft_landing_bonus | 0.013810 | 0.013810 | 0.015705 | 0.085266 |
| stability_penalty | -0.006025 | 0.006025 | 1.000000 | -0.037200 |
| total_reward | 0.164814 | 0.176646 | 1.000000 | 1.017610 |
| generated_reward | 0.164814 | 0.176646 | 1.000000 | 1.017610 |
| original_env_reward | -1.722980 | 2.425202 | 1.000000 | -10.638209 |

## Distribution
- score: mean=-118.437388, min=-140.549214, max=-105.349835
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -110.68 | -110.68 | 0.00 | 72.00 | energy_penalty=-0.008 progress_delta_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.242 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -118.44 | -110.68 | -7.75 | 71.90 | energy_penalty=-0.005 progress_delta_reward=0.162 soft_landing_bonus=0.014 stability_penalty=-0.006 | no_meaningful_improvement |
