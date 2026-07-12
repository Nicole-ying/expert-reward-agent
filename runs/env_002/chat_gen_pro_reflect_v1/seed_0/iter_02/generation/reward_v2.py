def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.0 * forward_velocity

    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    angle_penalty: hinge 惩罚，阈值降至 0.2 rad (~11.5°)
    #    系数 -1.0 使 per‑step 惩罚与前进奖励尺度匹配
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.2
    angle_penalty = -1.0 * max(0.0, abs(hull_angle) - angle_threshold)

    angular_vel_penalty = -0.1 * hull_angular_vel ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. soft_health_gate: 保持原状
    # ============================================================
    gate_k = 3.0
    health_gate = 1.0 / (1.0 + gate_k * abs(hull_angle))
    gated_forward_reward = forward_reward * health_gate

    # ============================================================
    # 4. 组合
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty

    components = {
        'forward_reward': forward_reward,
        'health_gate': health_gate,
        'gated_forward_reward': gated_forward_reward,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty
    }

    return float(total_reward), components