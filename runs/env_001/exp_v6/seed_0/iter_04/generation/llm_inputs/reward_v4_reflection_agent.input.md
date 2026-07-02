# 上一轮奖励函数代码（该轮得分: 142.675511）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    #    上轮 mean=0.004 过弱，导致 soft_landing_proxy (ratio=51.76) 主导总奖励。
    #    本轮加 scale=10（skeleton 建议 5~20），预计 progress mean ~0.04，
    #    使 soft/progress ratio 从 52 降至 ~5，恢复 progress 的主导地位。
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0  # 加 scale，每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（本轮未改，留待下轮观察）
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

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（系数降低 10 倍：0.01 → 0.001）
    #    原因：上轮 ratio_to_progress = -0.88，惩罚几乎抵消全部进度信号，
    #    导致 agent 无有效梯度可学。削弱后预计 ratio ≈ -0.09，progress 将主导。
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（本轮未改，留待下轮观察）
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
score=142.675511, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.045813 | 0.048503 | 0.998634 | 1.000000 |
| soft_landing_proxy | 0.220512 | 0.220512 | 0.441024 | 4.813341 |
| stability_penalty | -0.000613 | 0.000613 | 1.000000 | -0.013382 |
| total_reward | 0.265711 | 0.268230 | 1.000000 | 5.799959 |
| generated_reward | 0.265711 | 0.268230 | 1.000000 | 5.799959 |
| original_env_reward | -0.193894 | 1.740157 | 1.000000 | -4.232325 |

## Distribution
- score: mean=142.675511, min=120.986523, max=175.143642
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -112.37 | -112.37 | 0.00 | 74.10 | progress_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | 141.58 | 141.58 | 0.00 | 1000.00 | progress_reward=0.004 soft_landing_proxy=0.214 stability_penalty=-0.000 | new_best |
| 3 | progress_reward + soft_landing_proxy + stability_penalty | 142.68 | 141.58 | 1.10 | 1000.00 | progress_reward=0.046 soft_landing_proxy=0.221 stability_penalty=-0.001 | no_meaningful_improvement |
