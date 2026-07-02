# 上一轮奖励函数代码（该轮得分: 147.941570）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # reward = Φ(next_obs) - Φ(obs) = sum of improvements across all dimensions.
    # This is unchanged from the successful previous iteration.

    # --- positions ---
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5

    # --- speeds (L1 norm) ---
    speed_obs = abs(obs[2]) + abs(obs[3])
    speed_next = abs(next_obs[2]) + abs(next_obs[3])

    # --- angle ---
    angle_obs = abs(obs[4])
    angle_next = abs(next_obs[4])

    # --- angular velocity ---
    angvel_obs = abs(obs[5])
    angvel_next = abs(next_obs[5])

    # --- potential values (unchanged weights) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    # --- reward from potential improvement ---
    potential_reward = phi_next - phi_obs

    # --- landing incentive: continuous multiplicative signal ---
    # Only gives meaningful reward when near AND slow AND upright simultaneously.
    # Uses bounded 1/(1+kx) form on next_obs so it provides gradient toward good landing states.
    near_factor = 1.0 / (1.0 + 3.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 2.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 3.0 * angle_next)
    landing_incentive = 0.5 * near_factor * slow_factor * upright_factor

    # --- total ---
    total_reward = potential_reward + landing_incentive

    # --- decompose for logging ---
    progress_delta = dist_obs - dist_next
    speed_reduction = w_speed * (speed_obs - speed_next)
    angle_reduction = w_angle * (angle_obs - angle_next)
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)

    components = {
        'progress_delta': progress_delta,
        'speed_reduction': speed_reduction,
        'angle_reduction': angle_reduction,
        'angvel_reduction': angvel_reduction,
        'landing_incentive': landing_incentive,
        'total_reward': total_reward,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=147.941570, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angle_reduction | -0.000024 | 0.000136 | 0.999658 | -0.012824 |
| angvel_reduction | -0.000024 | 0.000291 | 0.999249 | -0.012688 |
| landing_incentive | 0.297907 | 0.297907 | 1.000000 | 156.782962 |
| progress_delta | 0.001900 | 0.002233 | 0.999368 | 1.000000 |
| speed_reduction | 0.000021 | 0.000699 | 1.000000 | 0.011282 |
| total_reward | 0.299780 | 0.299807 | 1.000000 | 157.768732 |
| generated_reward | 0.299780 | 0.299807 | 1.000000 | 157.768732 |
| original_env_reward | 0.010160 | 1.668054 | 1.000000 | 5.347179 |

## Distribution
- score: mean=147.941570, min=122.041022, max=181.904406
- episode_length: mean=1000.000000
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
| 5 | angle_reduction + angvel_reduction + landing_incentive + progress_delta + speed_reduction | 147.94 | 147.94 | 0.00 | 1000.00 | angle_reduction=-0.000 angvel_reduction=-0.000 landing_incentive=0.298 progress_delta=0.002 speed_reduction=0.000 | new_best |
