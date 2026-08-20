def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 信号提取 ──
    body_z_next      = next_obs[0]
    qx_next          = next_obs[2]
    qy_next          = next_obs[3]
    body_x_vel_next  = next_obs[13]
    body_y_vel_next  = next_obs[14]

    # ── 健康因子：高度 ──
    # 终止边界：[0.2, 1.0]；安全区 [0.3, 0.9]
    if body_z_next < 0.3:
        height_factor = (body_z_next - 0.2) / 0.1   # 在 z=0.2 时为 0
    elif body_z_next > 0.9:
        height_factor = (1.0 - body_z_next) / 0.1   # 在 z=1.0 时为 0
    else:
        height_factor = 1.0
    height_factor = max(0.0, min(1.0, height_factor))  # 允许到 0

    # ── 健康因子：直立度 ──
    body_up_z = 1.0 - 2.0 * (qx_next**2 + qy_next**2)  # 1=完全直立, -1=倒立
    # 安全阈值 0.3（约 17° 倾斜），在 0.3→-0.2 线性衰减
    upright_factor = (body_up_z + 0.2) / 0.5           # 在 -0.2 时为 0
    upright_factor = max(0.0, min(1.0, upright_factor))

    # ── 健康门控（min-of-hinge） ──
    health_gate = min(height_factor, upright_factor)

    # ── 前进奖励（gated） ──
    forward_speed = max(0.0, body_x_vel_next)
    progress_reward = 2.0 * (forward_speed ** 2) * health_gate

    # ── 动作效率惩罚 ──
    action_penalty = -0.02 * sum(a_i ** 2 for a_i in action)

    # ── 侧向漂移惩罚 ──
    lateral_penalty = -0.1 * (body_y_vel_next ** 2)

    total_reward = progress_reward + action_penalty + lateral_penalty

    components = {
        'progress': progress_reward,
        'action_penalty': action_penalty,
        'lateral_penalty': lateral_penalty,
        'health_gate': health_gate
    }
    return float(total_reward), components