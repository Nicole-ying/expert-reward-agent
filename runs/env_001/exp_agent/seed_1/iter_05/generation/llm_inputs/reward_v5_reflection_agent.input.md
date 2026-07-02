# 上一轮奖励函数代码（该轮得分: 235.963455）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：bounded proximity（不变） ===
    x_next, y_next = next_obs[0], next_obs[1]
    d_next = (x_next ** 2 + y_next ** 2) ** 0.5
    k_proximity = 5.0
    bounded_proximity = 1.0 / (1.0 + k_proximity * d_next)

    # === 稳定约束（不变） ===
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004
    w_angle = 0.02
    w_angvel = 0.004
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # === 软着陆奖励：二值 → 连续乘积（核心改动） ===
    # 每个因子用 max(0, 1-x/D) 提供平滑梯度，阈值 0.2（原二值阈值 0.1 的双倍）
    near_factor = max(0.0, 1.0 - d_next / 0.2)
    speed = abs(x_vel) + abs(y_vel)
    slow_factor = max(0.0, 1.0 - speed / 0.2)
    level_factor = max(0.0, 1.0 - abs(body_angle) / 0.2)
    # 足部接触：均值替代 min，单脚着地也给部分信号（0→0, 单脚→0.5, 双脚→1）
    feet_factor = 0.5 * (next_obs[6] + next_obs[7])

    soft_landing_raw = near_factor * slow_factor * level_factor * feet_factor
    soft_landing_bonus = 0.5 * soft_landing_raw  # 权重保持 0.5

    total_reward = bounded_proximity - stability_penalty + soft_landing_bonus

    components = {
        "bounded_proximity": bounded_proximity,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：bounded proximity ===
    # 替代 progress_delta。1/(1+k*dist) 自动 bounded 在 [0,1]，
    # 靠近目标时自然增长，始终为正（鼓励存活），提供平滑梯度。
    x_next, y_next = next_obs[0], next_obs[1]
    d_next = (x_next ** 2 + y_next ** 2) ** 0.5
    k_proximity = 5.0  # k=5: dist=1→0.167, dist=0.5→0.286, dist=0.1→0.667
    bounded_proximity = 1.0 / (1.0 + k_proximity * d_next)

    # === 稳定约束（保持上轮轻量权重） ===
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004
    w_angle = 0.02
    w_angvel = 0.004
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # === 软着陆奖励（保持原有逻辑） ===
    near = (d_next < 0.1)
    slow = (abs(x_vel) < 0.1 and abs(y_vel) < 0.1)
    level = (abs(body_angle) < 0.1)
    both_feet = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_bonus_raw = 1.0 if (near and slow and level and both_feet) else 0.0
    soft_landing_bonus = 0.5 * soft_landing_bonus_raw

    total_reward = bounded_proximity - stability_penalty + soft_landing_bonus

    components = {
        "bounded_proximity": bounded_proximity,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=235.963455, len=478.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| bounded_proximity | 0.559019 | 0.559019 | 1.000000 | 0.559019 |
| soft_landing_bonus | 0.156168 | 0.156168 | 0.512087 | 0.156168 |
| stability_penalty | 0.003080 | 0.003080 | 1.000000 | 0.003080 |
| total_reward | 0.712108 | 0.712108 | 1.000000 | 0.712108 |
| generated_reward | 0.712108 | 0.712108 | 1.000000 | 0.712108 |
| original_env_reward | -0.006798 | 1.513236 | 1.000000 | -0.006798 |

## Distribution
- score: mean=235.963455, min=161.961398, max=283.077842
- episode_length: mean=478.900000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_bonus + stability_penalty | -108.94 | -108.94 | 0.00 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.147 | new_best |
| 2 | progress_delta + soft_landing_bonus + stability_penalty | -110.57 | -108.94 | -1.62 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.006 | no_meaningful_improvement |
| 3 | bounded_proximity + soft_landing_bonus + stability_penalty | 251.94 | 251.94 | 0.00 | 363.30 | bounded_proximity=0.560 soft_landing_bonus=0.178 stability_penalty=0.003 | target_solved_new_best |
| 4 | bounded_proximity + soft_landing_bonus + stability_penalty | 235.96 | 251.94 | -15.97 | 478.90 | bounded_proximity=0.559 soft_landing_bonus=0.156 stability_penalty=0.003 | target_solved_no_improvement |
