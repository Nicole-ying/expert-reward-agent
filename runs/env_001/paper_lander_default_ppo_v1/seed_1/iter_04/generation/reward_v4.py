def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0

    # ---- 稳定/安全约束：轻量级惩罚 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    stability_penalty_value = (
        0.01 * abs(vx)
        + 0.01 * abs(vy)
        + 0.01 * abs(angle)
        + 0.005 * abs(angular_vel)
    )
    stability_penalty = -stability_penalty_value

    # ---- 连续着陆逼近信号：几何平均替代乘积，防止塌缩 ----
    distance_to_pad = next_distance
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle_abs = abs(angle)

    proximity_score = max(0.0, 1.0 - distance_to_pad / 1.0)
    speed_score = max(0.0, 1.0 - speed / 0.5)
    angle_score = max(0.0, 1.0 - angle_abs / 0.3)

    # 几何平均：保留AND语义（任一为零则整体为零），但对中等值不再塌缩
    soft_landing_bonus = (proximity_score * speed_score * angle_score) ** (1.0 / 3.0) * 2.0

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components