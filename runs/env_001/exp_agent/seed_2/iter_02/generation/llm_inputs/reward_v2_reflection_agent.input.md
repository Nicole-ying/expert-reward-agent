# 上一轮奖励函数代码（该轮得分: -105.897031）
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

    # ── 3. 轻量稳定约束 ──
    stability_penalty = -0.01 * (abs(vel_x) + abs(vel_y)) \
                        -0.01 * abs(angle) \
                        -0.01 * abs(angular_vel)

    # ── 4. 软着陆近似信号 ──
    # 当飞行器满足“靠近目标、低速、姿态竖直、双腿接触”时给予小额奖励
    near_target = dist_new < 0.15
    low_speed = (vel_x ** 2 + vel_y ** 2) ** 0.5 < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.5 if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    # ── 组件字典 ──
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
score=-105.897031, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.015973 | 0.016913 | 0.999995 | 0.015973 |
| soft_landing_proxy | 0.001721 | 0.001721 | 0.003442 | 0.001721 |
| stability_penalty | -0.014131 | 0.014131 | 1.000000 | -0.014131 |
| total_reward | 0.003563 | 0.009544 | 1.000000 | 0.003563 |
| generated_reward | 0.003563 | 0.009544 | 1.000000 | 0.003563 |
| original_env_reward | -1.521290 | 2.439374 | 1.000000 | -1.521290 |

## Distribution
- score: mean=-105.897031, min=-122.962817, max=-96.298785
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -105.90 | -105.90 | 0.00 | 72.00 | progress=0.016 soft_landing_proxy=0.002 stability_penalty=-0.014 | new_best |
