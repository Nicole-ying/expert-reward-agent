# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_001\exp_v6\seed_7\iter_03\generation\validations\reward_v3.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -110.220087）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：系数降低约10倍，使 penalty 不再主导 progress
    # 原系数 0.05/0.1/0.01 → ratio -4.24，agent 不敢动
    # 新系数 0.005/0.01/0.001 → 预期 ratio ~0.4，低于 0.5 安全线
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.005 * (abs(vel_x) + abs(vel_y)) - 0.01 * abs(angle) - 0.001 * abs(angular_vel)

    # 软着陆 proxy：当接近目标、低速、姿态小且双腿接触时给予一次性小奖励
    landing_bonus = 0.0
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (dist_next < 0.1 and
        (vel_x**2 + vel_y**2) ** 0.5 < 0.2 and
        abs(angle) < 0.2 and
        left_contact and right_contact):
        landing_bonus = 1.0

    total_reward = progress_delta + stability_penalty + landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：对高速、大角度、快角速度的轻量惩罚
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    # 各系数较小，避免抑制正常机动
    stability_penalty = -0.05 * (abs(vel_x) + abs(vel_y)) - 0.1 * abs(angle) - 0.01 * abs(angular_vel)

    # 软着陆 proxy：当接近目标、低速、姿态小且双腿接触时给予一次性小奖励
    landing_bonus = 0.0
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (dist_next < 0.1 and
        (vel_x**2 + vel_y**2) ** 0.5 < 0.2 and
        abs(angle) < 0.2 and
        left_contact and right_contact):
        landing_bonus = 1.0

    total_reward = progress_delta + stability_penalty + landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-110.220087, len=68.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_bonus | 0.001969 | 0.001969 | 0.001969 | 0.122030 |
| progress_delta | 0.016136 | 0.017056 | 0.999997 | 1.000000 |
| stability_penalty | -0.007052 | 0.007052 | 1.000000 | -0.437040 |
| total_reward | 0.011053 | 0.013441 | 1.000000 | 0.684990 |
| generated_reward | 0.011053 | 0.013441 | 1.000000 | 0.684990 |
| original_env_reward | -1.598682 | 2.436073 | 1.000000 | -99.075975 |

## Distribution
- score: mean=-110.220087, min=-126.217237, max=-99.842818
- episode_length: mean=68.200000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress_delta + stability_penalty | -99.87 | -99.87 | 0.00 | 68.30 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.068 | new_best |
| 2 | landing_bonus + progress_delta + stability_penalty | -110.22 | -99.87 | -10.35 | 68.20 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.007 | no_meaningful_improvement |
