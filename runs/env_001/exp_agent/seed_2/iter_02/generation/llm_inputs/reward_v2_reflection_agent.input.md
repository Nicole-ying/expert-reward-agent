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
score=-110.683396, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.008435 | 0.008435 | 0.168709 | -0.052693 |
| progress_delta_reward | 0.160088 | 0.169472 | 0.999990 | 1.000000 |
| soft_landing_bonus | 0.011517 | 0.011517 | 0.005759 | 0.071945 |
| stability_penalty | -0.242176 | 0.242176 | 1.000000 | -1.512771 |
| total_reward | -0.079006 | 0.105815 | 1.000000 | -0.493519 |
| generated_reward | -0.079006 | 0.105815 | 1.000000 | -0.493519 |
| original_env_reward | -1.545106 | 2.316389 | 1.000000 | -9.651624 |

## Distribution
- score: mean=-110.683396, min=-130.746612, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -110.68 | -110.68 | 0.00 | 72.00 | energy_penalty=-0.008 progress_delta_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.242 | new_best |
