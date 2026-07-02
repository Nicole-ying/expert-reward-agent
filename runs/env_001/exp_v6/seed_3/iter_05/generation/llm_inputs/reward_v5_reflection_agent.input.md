# 上一轮奖励函数代码（该轮得分: 84.797051）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # reward = Φ(next_obs) - Φ(obs) = sum of improvements across all dimensions.
    # This replaces the penalty paradigm: instead of punishing bad states,
    # we reward movement toward better states in every dimension simultaneously.

    # --- positions ---
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5

    # --- speeds (L1 norm for simplicity) ---
    speed_obs = abs(obs[2]) + abs(obs[3])
    speed_next = abs(next_obs[2]) + abs(next_obs[3])

    # --- angle ---
    angle_obs = abs(obs[4])
    angle_next = abs(next_obs[4])

    # --- angular velocity ---
    angvel_obs = abs(obs[5])
    angvel_next = abs(next_obs[5])

    # --- potential values ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    # --- reward: improvement in potential ---
    total_reward = phi_next - phi_obs

    # --- decompose for logging ---
    progress_delta = dist_obs - dist_next  # distance improvement
    speed_reduction = w_speed * (speed_obs - speed_next)  # speed improvement
    angle_reduction = w_angle * (angle_obs - angle_next)  # angle improvement
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)  # angvel improvement

    components = {
        'progress_delta': progress_delta,
        'speed_reduction': speed_reduction,
        'angle_reduction': angle_reduction,
        'angvel_reduction': angvel_reduction,
        'total_reward': total_reward,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=84.797051, len=823.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angle_reduction | -0.000084 | 0.000247 | 0.999782 | -0.020247 |
| angvel_reduction | -0.000102 | 0.000402 | 0.999368 | -0.024506 |
| progress_delta | 0.004144 | 0.004387 | 0.999548 | 1.000000 |
| speed_reduction | 0.000029 | 0.000826 | 1.000000 | 0.007020 |
| total_reward | 0.003988 | 0.004483 | 0.999998 | 0.962267 |
| generated_reward | 0.003988 | 0.004483 | 0.999998 | 0.962267 |
| original_env_reward | -0.129463 | 1.699018 | 1.000000 | -31.238671 |

## Distribution
- score: mean=84.797051, min=48.121934, max=126.608144
- episode_length: mean=823.300000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + stability_penalty | -112.57 | -112.57 | 0.00 | 70.60 | progress_delta=0.016 stability_penalty=-0.014 | new_best |
| 2 | progress_delta + stability_penalty | -96.19 | -96.19 | 0.00 | 99.80 | progress_delta=0.015 stability_penalty=-0.001 | new_best |
| 3 | progress_delta + stability_penalty | -101.92 | -96.19 | -5.73 | 86.80 | progress_delta=0.014 stability_penalty=-0.007 | no_meaningful_improvement |
| 4 | angle_reduction + angvel_reduction + progress_delta + speed_reduction | 84.80 | 84.80 | 0.00 | 823.30 | angle_reduction=-0.000 angvel_reduction=-0.000 progress_delta=0.004 speed_reduction=0.000 | new_best |
