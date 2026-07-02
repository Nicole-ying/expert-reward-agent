# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_001\exp_v6\seed_3\iter_03\generation\validations\reward_v3.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -96.193089）
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
score=-96.193089, len=99.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.014528 | 0.015154 | 0.999998 | 1.000000 |
| stability_penalty | -0.001433 | 0.001433 | 1.000000 | -0.098625 |
| total_reward | 0.013095 | 0.013868 | 1.000000 | 0.901375 |
| generated_reward | 0.013095 | 0.013868 | 1.000000 | 0.901375 |
| original_env_reward | -0.808494 | 3.186441 | 1.000000 | -55.650080 |

## Distribution
- score: mean=-96.193089, min=-123.567649, max=-36.964139
- episode_length: mean=99.800000
- early_terminal (<150 steps + score<-50): 9/10 (90%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + stability_penalty | -112.57 | -112.57 | 0.00 | 70.60 | progress_delta=0.016 stability_penalty=-0.014 | new_best |
| 2 | progress_delta + stability_penalty | -96.19 | -96.19 | 0.00 | 99.80 | progress_delta=0.015 stability_penalty=-0.001 | new_best |
