# 上一轮奖励函数代码（该轮得分: -81.524182）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 诊断与修改理由 ===
    # 问题：stability_penalty (-0.068) 是 shaping_reward (0.026) 的 2.6 倍，
    #       agent 在任何状态下都被净惩罚，导致 episode 短 (71步)、全部 crash。
    # 修改1：stability 系数降低 10x，使其成为弱背景信号（~27% of shaping）。
    # 修改2：gamma 1.0 消除静止时的 (1-γ)*dist 虚假奖励。
    # 修改3：components key 与公式变量名一致，移除 total_reward。

    # 1. 势能塑形：Φ = -distance, gamma = 1.0
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # gamma=1.0 → shaping = dist_current - dist_next，纯进度信号
    shaping_reward = dist_current - dist_next

    # 2. 稳定性惩罚：轻量背景约束
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # 系数降低 10x：原 0.05/0.02/0.05 → 现 0.005/0.002/0.005
    w_vel = 0.005
    w_ang = 0.002
    w_angle = 0.005

    stability_penalty = -(
        w_vel * (abs(vel_x) + abs(vel_y)) +
        w_ang * abs(ang_vel) +
        w_angle * abs(angle)
    )

    total_reward = shaping_reward + stability_penalty

    # components 只放总公式中直接出现的变量，key 与变量名一致
    components = {
        "shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 势能塑形：引导靠近目标点
    gamma = 0.99
    # 采用负欧几里得距离作为势能函数
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    shaping_reward = gamma * (-dist_next) - (-dist_current)  # = dist_current - gamma * dist_next

    # 2. 稳定性惩罚：基于下一时刻的速度、姿态、角速度
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    w_vel = 0.05        # 速度项系数
    w_ang = 0.02        # 角速度项系数
    w_angle = 0.05      # 姿态角系数

    stability_penalty = -(
        w_vel * (abs(vel_x) + abs(vel_y)) +
        w_ang * abs(ang_vel) +
        w_angle * abs(angle)
    )

    total_reward = shaping_reward + stability_penalty

    components = {
        "potential_shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-81.524182, len=98.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| shaping_reward | 0.014089 | 0.014816 | 0.999999 | 0.014089 |
| stability_penalty | -0.006081 | 0.006081 | 1.000000 | -0.006081 |
| total_reward | 0.008008 | 0.009973 | 1.000000 | 0.008008 |
| generated_reward | 0.008008 | 0.009973 | 1.000000 | 0.008008 |
| original_env_reward | -0.910745 | 2.758755 | 1.000000 | -0.910745 |

## Distribution
- score: mean=-81.524182, min=-119.011980, max=-13.808814
- episode_length: mean=98.100000
- early_terminal (<150 steps + score<-50): 8/10 (80%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -111.44 | -111.44 | 0.00 | 71.30 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.057 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | -111.12 | -111.44 | 0.33 | 71.40 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.006 | no_meaningful_improvement |
| 4 | potential_shaping_reward + stability_penalty | -111.89 | -111.89 | 0.00 | 71.30 | potential_shaping_reward=0.026 stability_penalty=-0.068 | new_best |
| 5 | shaping_reward + stability_penalty | -81.52 | -81.52 | 0.00 | 98.10 | shaping_reward=0.014 stability_penalty=-0.006 | new_best |
