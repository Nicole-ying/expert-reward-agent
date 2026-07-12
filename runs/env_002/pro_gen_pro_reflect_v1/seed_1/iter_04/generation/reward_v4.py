def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：温和地惩罚角速度，减少晃动 ----------
    w_ang_vel = 0.05  # 从0.1降至0.05，避免过度约束动态探索
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 动作效率：极轻惩罚动作力矩平方和，鼓励探索更高效步态 ----------
    w_action = 0.02  # 从0.1大幅降至0.02，避免抑制速度探索
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components