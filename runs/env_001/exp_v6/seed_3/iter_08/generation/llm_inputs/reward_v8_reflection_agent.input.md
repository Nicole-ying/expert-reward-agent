# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\exp_v6\seed_3\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 256.268428）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # Unchanged backbone from v6 — provides the main progress gradient.

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

    # --- potential values (weights unchanged from v6) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    potential_reward = phi_next - phi_obs

    # --- landing incentive: bounded proximity gating ---
    # Peak reduced 1.5→1.2 to ease the 236x dominance over progress signal,
    # while retaining strong terminal guidance for the learned landing behavior.
    near_factor = 1.0 / (1.0 + 15.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 8.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 8.0 * angle_next)
    landing_incentive = 1.2 * near_factor * slow_factor * upright_factor

    # --- total ---
    total_reward = potential_reward + landing_incentive

    # --- components: only variables directly in the total_reward formula ---
    components = {
        'potential_reward': potential_reward,
        'landing_incentive': landing_incentive,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # This backbone is unchanged — it provides the main progress gradient.

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

    potential_reward = phi_next - phi_obs

    # --- landing incentive: sharpened proximity gating ---
    # Old version used k_dist=3, k_speed=2, k_angle=3 — too wide, dominated progress.
    # New version uses much steeper decay so reward concentrates near true touchdown
    # (dist<0.3, speed<0.3, angle<0.15). Peak raised to 1.5 to maintain strong terminal gradient.
    near_factor = 1.0 / (1.0 + 15.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 8.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 8.0 * angle_next)
    landing_incentive = 1.5 * near_factor * slow_factor * upright_factor

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
score=256.268428, len=349.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_incentive | 0.428239 | 0.428239 | 1.000000 | 0.428239 |
| potential_reward | 0.002288 | 0.002802 | 0.999965 | 0.002288 |
| total_reward | 0.430527 | 0.430598 | 1.000000 | 0.430527 |
| generated_reward | 0.430527 | 0.430598 | 1.000000 | 0.430527 |
| original_env_reward | 0.020168 | 1.441309 | 1.000000 | 0.020168 |

## Distribution
- score: mean=256.268428, min=231.043827, max=293.109267
- episode_length: mean=349.600000
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
| 6 | angle_reduction + angvel_reduction + landing_incentive + progress_delta + speed_reduction | 260.64 | 260.64 | 0.00 | 314.40 | angle_reduction=-0.000 angvel_reduction=-0.000 landing_incentive=0.508 progress_delta=0.002 speed_reduction=0.000 | target_solved_new_best |
| 7 | landing_incentive + potential_reward | 256.27 | 260.64 | -4.37 | 349.60 | landing_incentive=0.428 potential_reward=0.002 | target_solved_no_improvement |
