def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度

    # ---------- 健康门控 ----------
    danger_angle = 0.8
    max_angle = 1.2
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：hinge 形态 ----------
    # 安全阈值 2.0：在此范围内不计入惩罚
    # 超出部分线性惩罚，为接近摔倒的高角速度提供梯度
    stability_threshold = 2.0
    excess_ang_vel = max(0.0, abs(hull_ang_vel_next) - stability_threshold)
    stability_penalty = -0.05 * excess_ang_vel

    # ---------- 动作效率 ----------
    w_action = 0.02
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + stability_penalty + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components