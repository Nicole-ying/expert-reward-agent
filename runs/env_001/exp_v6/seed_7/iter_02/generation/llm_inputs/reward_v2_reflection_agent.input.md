# 上一轮奖励函数代码（该轮得分: -99.873069）
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
score=-99.873069, len=68.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_bonus | 0.002098 | 0.002098 | 0.002098 | 0.130421 |
| progress_delta | 0.016083 | 0.017017 | 0.999990 | 1.000000 |
| stability_penalty | -0.068260 | 0.068260 | 1.000000 | -4.244113 |
| total_reward | -0.050079 | 0.054223 | 1.000000 | -3.113693 |
| generated_reward | -0.050079 | 0.054223 | 1.000000 | -3.113693 |
| original_env_reward | -1.506845 | 2.423347 | 1.000000 | -93.688979 |

## Distribution
- score: mean=-99.873069, min=-119.076316, max=-77.537854
- episode_length: mean=68.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress_delta + stability_penalty | -99.87 | -99.87 | 0.00 | 68.30 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.068 | new_best |
