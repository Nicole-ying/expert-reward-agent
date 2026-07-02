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