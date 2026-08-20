# 上一轮奖励函数代码（该轮得分: 124.965291）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励（保持不变）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 连续软着陆 proxy — 从二值改为 bounded 乘积，全程梯度 + 自动 anti-exploit
    # near_factor: 1/(1+10*dist), dist=0→1.0, dist=0.1→0.5, dist=0.5→0.17
    near_factor = 1.0 / (1.0 + 10.0 * next_dist)
    # speed_factor: 1/(1+5*speed), speed=0→1.0, speed=0.2→0.5, speed=0.5→0.29
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    # angle_factor: 1/(1+10*angle), angle=0→1.0, angle=0.1→0.5, angle=0.3→0.25
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    # contact_factor: 连续 0~1，两条腿平均接触
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.1 * near_factor * speed_factor * angle_factor * contact_factor

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（弱背景信号，ratio ≈ -0.014，不干扰主信号）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（二值事件型，iter 5 验证得分 146.13）
    near_target = (next_dist < 0.1)
    low_speed = (vel_x + vel_y < 0.2)
    stable_angle = (angle < 0.1)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_proxy = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=124.965291, len=919.400000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.003727 | 0.003973 | 0.999464 | 1.000000 |
| soft_landing_proxy | 0.046488 | 0.046488 | 0.635354 | 12.473117 |
| stability_penalty | -0.000463 | 0.000463 | 1.000000 | -0.124259 |
| total_reward | 0.049752 | 0.049980 | 1.000000 | 13.348858 |
| generated_reward | 0.049752 | 0.049980 | 1.000000 | 13.348858 |
| original_env_reward | -0.093255 | 1.465653 | 1.000000 | -25.020811 |

## Distribution
- score: mean=124.965291, min=-39.682863, max=179.800901
- episode_length: mean=919.400000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -112.37 | -112.37 | 0.00 | 74.10 | progress_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | 141.58 | 141.58 | 0.00 | 1000.00 | progress_reward=0.004 soft_landing_proxy=0.214 stability_penalty=-0.000 | new_best |
| 3 | progress_reward + soft_landing_proxy + stability_penalty | 142.68 | 141.58 | 1.10 | 1000.00 | progress_reward=0.046 soft_landing_proxy=0.221 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress_reward + soft_landing_proxy + stability_penalty | 66.52 | 141.58 | -75.06 | 1000.00 | progress_reward=0.045 soft_landing_proxy=0.077 stability_penalty=-0.001 | no_meaningful_improvement |
| 5 | progress_reward + soft_landing_proxy + stability_penalty | 146.13 | 141.58 | 4.55 | 1000.00 | progress_reward=0.045 soft_landing_proxy=0.206 stability_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 6 | progress_reward + soft_landing_proxy + stability_penalty | 141.58 | 141.58 | 0.00 | 1000.00 | progress_reward=0.004 soft_landing_proxy=0.214 stability_penalty=-0.000 | new_best |
| 7 | progress_reward + soft_landing_proxy + stability_penalty | 124.97 | 141.58 | -16.61 | 919.40 | progress_reward=0.004 soft_landing_proxy=0.046 stability_penalty=-0.000 | no_meaningful_improvement |
