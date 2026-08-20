# ⚠️ 上一版代码验证失败
错误信息：Reward v5 failed validation: runs\env_001\exp_v6\seed_4\iter_05\generation\validations\reward_v5.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -111.893442）
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
score=-111.893442, len=71.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| potential_shaping_reward | 0.025789 | 0.026064 | 1.000000 | 0.025789 |
| stability_penalty | -0.067973 | 0.067973 | 1.000000 | -0.067973 |
| total_reward | -0.042184 | 0.042488 | 1.000000 | -0.042184 |
| generated_reward | -0.042184 | 0.042488 | 1.000000 | -0.042184 |
| original_env_reward | -1.558004 | 2.418792 | 1.000000 | -1.558004 |

## Distribution
- score: mean=-111.893442, min=-123.576363, max=-96.699294
- episode_length: mean=71.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -111.44 | -111.44 | 0.00 | 71.30 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.057 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | -111.12 | -111.44 | 0.33 | 71.40 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.006 | no_meaningful_improvement |
| 4 | potential_shaping_reward + stability_penalty | -111.89 | -111.89 | 0.00 | 71.30 | potential_shaping_reward=0.026 stability_penalty=-0.068 | new_best |
