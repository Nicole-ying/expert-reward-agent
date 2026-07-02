# 上一轮奖励函数代码（该轮得分: 187.925005）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励 ──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（系数从 0.01 → 0.002，降低 5 倍）──
    # 上一轮：penalty ratio = -0.88，严重压制 progress 信号
    # 本轮目标：ratio ≈ -0.18，让 agent 敢于机动
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 软着陆近似信号（本轮不动）──
    near_target = dist_new < 0.15
    low_speed = (vel_x ** 2 + vel_y ** 2) ** 0.5 < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.5 if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=187.925005, len=694.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.002937 | 0.003230 | 0.999486 | 0.002937 |
| soft_landing_proxy | 0.257389 | 0.257389 | 0.514778 | 0.257389 |
| stability_penalty | -0.000804 | 0.000804 | 1.000000 | -0.000804 |
| total_reward | 0.259522 | 0.259994 | 1.000000 | 0.259522 |
| generated_reward | 0.259522 | 0.259994 | 1.000000 | 0.259522 |
| original_env_reward | -0.008679 | 1.454366 | 1.000000 | -0.008679 |

## Distribution
- score: mean=187.925005, min=98.774145, max=270.681596
- episode_length: mean=694.500000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -105.90 | -105.90 | 0.00 | 72.00 | progress=0.016 soft_landing_proxy=0.002 stability_penalty=-0.014 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | 187.93 | 187.93 | 0.00 | 694.50 | progress=0.003 soft_landing_proxy=0.257 stability_penalty=-0.001 | new_best |
