def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    hull_angle_next = next_obs[0]
    hull_ang_vel_next = next_obs[1]
    horizontal_velocity_next = next_obs[2]

    # ---------- 健康门控 ----------
    danger_angle = 0.8
    max_angle = 1.2
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 姿态稳定性：连续折扣 ----------
    safe_offset = 0.15
    excess_angle = max(0.0, abs(hull_angle_next) - safe_offset)
    stability_ratio = 1.0 / (1.0 + 25.0 * excess_angle ** 2)  # 安全区≈1, 危险区快速衰减
    stability_bonus = 0.5 * stability_ratio * horizontal_velocity_next

    # ---------- 动作效率 ----------
    w_action = 0.02
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + stability_bonus + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_bonus": stability_bonus,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components