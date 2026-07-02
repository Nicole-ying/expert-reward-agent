# ⚠️ 上一版代码验证失败
错误信息：Reward v10 failed validation: runs\env_001\exp_v6\seed_0\iter_10\generation\validations\reward_v10.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 149.707807）
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

    # 3. 连续软着陆 proxy — 从乘积改为接触门控加权和
    # 原因：乘积形态下 ∂(a·b·c)/∂a = b·c，任意因子小时所有梯度坍缩。
    # 和形态下每个姿势因子有独立梯度，contact_factor 作为全局门控保留 AND 约束。
    # 系数从 0.05 降至 0.01 补偿和形式天然更大的量级（三个 ~0.5 因子之和 ≈1.5，而乘积 ≈0.12）。
    dist_gate = 1.0 / (1.0 + 100.0 * next_dist ** 2)
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.01 * contact_factor * (dist_gate + speed_factor + angle_factor)

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

    # 3. 连续软着陆 proxy — 尖锐距离门控 + 连续梯度
    # dist_gate: 1/(1+100*dist^2), dist=0.1→0.5, dist=0.2→0.2, dist=0.3→0.1
    # 只在真正接近目标时激活，避免远距离 proxy hacking
    dist_gate = 1.0 / (1.0 + 100.0 * next_dist ** 2)
    # speed_factor: 1/(1+5*speed), speed=0→1.0, speed=0.2→0.5
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    # angle_factor: 1/(1+10*angle), angle=0→1.0, angle=0.1→0.5
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    # contact_factor: 连续 0~1，两条腿平均接触
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.05 * dist_gate * speed_factor * angle_factor * contact_factor

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
score=149.707807, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.003376 | 0.003625 | 0.999529 | 1.000000 |
| soft_landing_proxy | 0.017953 | 0.017953 | 0.667874 | 5.317245 |
| stability_penalty | -0.000432 | 0.000432 | 1.000000 | -0.127827 |
| total_reward | 0.020898 | 0.021111 | 1.000000 | 6.189418 |
| generated_reward | 0.020898 | 0.021111 | 1.000000 | 6.189418 |
| original_env_reward | -0.074682 | 1.492816 | 1.000000 | -22.119176 |

## Distribution
- score: mean=149.707807, min=125.984617, max=182.024049
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
| 6 | progress_reward + soft_landing_proxy + stability_penalty | 141.58 | 141.58 | 0.00 | 1000.00 | progress_reward=0.004 soft_landing_proxy=0.214 stability_penalty=-0.000 | new_best |
| 7 | progress_reward + soft_landing_proxy + stability_penalty | 124.97 | 141.58 | -16.61 | 919.40 | progress_reward=0.004 soft_landing_proxy=0.046 stability_penalty=-0.000 | no_meaningful_improvement |
| 8 | progress_reward + soft_landing_proxy + stability_penalty | 152.38 | 152.38 | 0.00 | 1000.00 | progress_reward=0.004 soft_landing_proxy=0.026 stability_penalty=-0.000 | new_best |
| 9 | progress_reward + soft_landing_proxy + stability_penalty | 149.71 | 152.38 | -2.67 | 1000.00 | progress_reward=0.003 soft_landing_proxy=0.018 stability_penalty=-0.000 | no_meaningful_improvement |
