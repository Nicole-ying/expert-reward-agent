def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle = next_obs[0]          # body pitch angle
    hull_angular_vel = next_obs[1]   # body angular velocity (rad/s)  ← 新使用
    horizontal_speed = next_obs[2]   # forward velocity
    vertical_speed = next_obs[3]     # vertical velocity
    leg1_contact = next_obs[12]      # left leg ground contact (0 or 1)
    leg2_contact = next_obs[13]      # right leg ground contact (0 or 1)

    # 主进展信号：水平速度
    progress_raw = horizontal_speed

    # 身体倾角门控（保持原有）
    angle_threshold = 0.6
    angle_gate = max(0.0, 1.0 - abs(hull_angle) / angle_threshold)

    # 垂直弹跳门控（保持原有）
    vert_threshold = 2.0
    vertical_gate = max(0.0, 1.0 - abs(vertical_speed) / vert_threshold)

    # 综合健康门控（几何平均，保持原有）
    health_gate = ((angle_gate * vertical_gate) + 1e-8) ** 0.5

    # 基础奖励：进度 × 健康
    base_reward = progress_raw * health_gate

    # ── 新增：摔倒风险惩罚 ──
    # (1) 角速度惩罚：超过 1.0 rad/s 的部分线性惩罚，系数 0.05
    angular_vel_penalty = max(0.0, abs(hull_angular_vel) - 1.0) * 0.05

    # (2) 双脚离地惩罚：当两腿均未触地时给予轻度惩罚，系数 0.03
    #     使用连续乘积避免二值突变： (1-leg1)*(1-leg2) 仅在双脚离地时 ≈1
    air_penalty = (1.0 - leg1_contact) * (1.0 - leg2_contact) * 0.03

    falling_risk_penalty = angular_vel_penalty + air_penalty

    total_reward = base_reward - falling_risk_penalty

    components = {
        'progress_raw': progress_raw,
        'angle_gate': angle_gate,
        'vertical_gate': vertical_gate,
        'health_gate': health_gate,
        'falling_risk_penalty': falling_risk_penalty,
    }

    return float(total_reward), components