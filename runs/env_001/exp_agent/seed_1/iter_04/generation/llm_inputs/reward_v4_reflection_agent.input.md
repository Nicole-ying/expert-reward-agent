# 上一轮奖励函数代码（该轮得分: 251.935107）
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
score=251.935107, len=363.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| bounded_proximity | 0.560437 | 0.560437 | 1.000000 | 0.560437 |
| soft_landing_bonus | 0.177533 | 0.177533 | 0.355065 | 0.177533 |
| stability_penalty | 0.003076 | 0.003076 | 1.000000 | 0.003076 |
| total_reward | 0.734895 | 0.734895 | 1.000000 | 0.734895 |
| generated_reward | 0.734895 | 0.734895 | 1.000000 | 0.734895 |
| original_env_reward | -0.036029 | 2.021793 | 1.000000 | -0.036029 |

## Distribution
- score: mean=251.935107, min=214.172774, max=283.009041
- episode_length: mean=363.300000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_bonus + stability_penalty | -108.94 | -108.94 | 0.00 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.147 | new_best |
| 2 | progress_delta + soft_landing_bonus + stability_penalty | -110.57 | -108.94 | -1.62 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.006 | no_meaningful_improvement |
| 3 | bounded_proximity + soft_landing_bonus + stability_penalty | 251.94 | 251.94 | 0.00 | 363.30 | bounded_proximity=0.560 soft_landing_bonus=0.178 stability_penalty=0.003 | target_solved_new_best |
