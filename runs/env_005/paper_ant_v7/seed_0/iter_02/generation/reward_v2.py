def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 高度安全门控 ----------
    # [0.2, 0.3] 线性 0→1, [0.9, 1.0] 线性 1→0
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)  # 安全区=1, 越危险→0

    # ---------- 主学习信号：被高度门控的前进速度 ----------
    forward_reward = 1.0 * body_x_velocity * height_factor

    # ---------- 直立姿态约束（二次惩罚） ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    upright_penalty = -2.0 * (upright_error**2)

    # ---------- 力矩效率约束 ----------
    action_cost = -0.01 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components