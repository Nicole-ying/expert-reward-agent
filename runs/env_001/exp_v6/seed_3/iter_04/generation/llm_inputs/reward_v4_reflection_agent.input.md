# 上一轮奖励函数代码（该轮得分: -101.922514）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- progress signal: reduction in Euclidean distance to target ----
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # >0 when moving toward target

    # ---- distance-gated stability penalty ----
    # Gate: 1/(1+dist) → near 0 when far (free to maneuver),
    #                     near 1 when at target (enforce fine stability).
    # This creates a natural curriculum: aggressive far, careful near.
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    gate = 1.0 / (1.0 + dist_next)

    # Base weights at iter-1 levels; distance gating auto-scales them down
    # when far from target, preventing the penalty from dominating progress.
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005

    stability_penalty = -gate * (
        w_vel * (abs(vx) + abs(vy))
        + w_angle * abs(angle)
        + w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- main progress signal: reduction in Euclidean distance to target ----
    # target position is (0, 0) in the relative coordinate system
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # positive when moving closer

    # ---- light stability penalty on next observation ----
    # penalise linear velocity, body angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # weights reduced 10x from previous iteration (were 0.01, 0.01, 0.005)
    # previous ratio_to_progress was -0.85 — penalty dominated progress entirely
    w_vel = 0.001
    w_angle = 0.001
    w_angvel = 0.0005

    stability_penalty = (
        - w_vel * (abs(vx) + abs(vy))
        - w_angle * abs(angle)
        - w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    # ---- component logging ----
    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-101.922514, len=86.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.014126 | 0.014905 | 0.999996 | 1.000000 |
| stability_penalty | -0.007117 | 0.007117 | 1.000000 | -0.503847 |
| total_reward | 0.007009 | 0.009615 | 1.000000 | 0.496153 |
| generated_reward | 0.007009 | 0.009615 | 1.000000 | 0.496153 |
| original_env_reward | -0.940239 | 2.703796 | 1.000000 | -66.560087 |

## Distribution
- score: mean=-101.922514, min=-124.339260, max=-78.190419
- episode_length: mean=86.800000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + stability_penalty | -112.57 | -112.57 | 0.00 | 70.60 | progress_delta=0.016 stability_penalty=-0.014 | new_best |
| 2 | progress_delta + stability_penalty | -96.19 | -96.19 | 0.00 | 99.80 | progress_delta=0.015 stability_penalty=-0.001 | new_best |
| 3 | progress_delta + stability_penalty | -101.92 | -96.19 | -5.73 | 86.80 | progress_delta=0.014 stability_penalty=-0.007 | no_meaningful_improvement |
