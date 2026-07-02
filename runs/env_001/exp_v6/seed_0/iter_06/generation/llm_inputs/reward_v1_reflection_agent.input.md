# ⚠️ 上一版代码验证失败
错误信息：Reward v1 failed validation: runs\env_001\exp_v6\seed_0\iter_06\generation\validations\reward_v1.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 66.522094）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：进度奖励（保持 scale=10，iter 3-4 均验证有效）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0

    # 2. 稳定性惩罚（保持不变，ratio=-0.014 已是无害背景信号）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy：从事件型回到每步连续型
    #    诊断：事件型 (just_landed) 使着陆激励缩水 65%，得分暴跌。
    #    修复：用 bounded 连续因子 max(0, 1-x/D) 替代 binary if-else，
    #    每个着陆条件独立贡献梯度，奖励随质量连续缩放。
    #    contact 也用连续值（乘积）替代 binary gate，为双脚接近地面提供梯度。

    near_factor = max(0.0, 1.0 - next_dist / 0.3)
    speed = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.2)

    # 连续接触因子：双脚乘积，提供"双脚同时着地"的梯度
    left_contact = min(1.0, max(0.0, next_obs[6]))
    right_contact = min(1.0, max(0.0, next_obs[7]))
    contact_factor = left_contact * right_contact

    # 四个因子相乘：任何维度偏离最佳状态都会降低奖励
    soft_landing_proxy = near_factor * speed_factor * angle_factor * contact_factor * 0.5

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
score=66.522094, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.045367 | 0.048510 | 0.999966 | 1.000000 |
| soft_landing_proxy | 0.077214 | 0.077214 | 0.077214 | 1.701983 |
| stability_penalty | -0.000628 | 0.000628 | 1.000000 | -0.013850 |
| total_reward | 0.121953 | 0.125075 | 1.000000 | 2.688133 |
| generated_reward | 0.121953 | 0.125075 | 1.000000 | 2.688133 |
| original_env_reward | -0.250546 | 4.386100 | 1.000000 | -5.522627 |

## Distribution
- score: mean=66.522094, min=40.938003, max=95.685433
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
| 4 | progress_reward + soft_landing_proxy + stability_penalty | 66.52 | 141.58 | -75.06 | 1000.00 | progress_reward=0.045 soft_landing_proxy=0.077 stability_penalty=-0.001 | no_meaningful_improvement |
| 5 | progress_reward + soft_landing_proxy + stability_penalty | 146.13 | 141.58 | 4.55 | 1000.00 | progress_reward=0.045 soft_landing_proxy=0.206 stability_penalty=-0.001 | unsolved_stagnation_fresh_restart |
