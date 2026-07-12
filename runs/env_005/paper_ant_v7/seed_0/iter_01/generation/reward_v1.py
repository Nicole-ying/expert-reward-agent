def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    # 注意：action 是 8 维力矩，范围 [-1, 1]
    torque_penalty = sum([a**2 for a in action])  # 默认可迭代，但保险起见写成列表

    # ---------- 主学习信号：向前速度 ----------
    # 线性正奖励，鼓励更大的向前速度
    forward_reward = 1.0 * body_x_velocity

    # ---------- 身体高度生存约束（hinge 惩罚） ----------
    # 健康区间为 [0.3, 0.9]，接近边界 0.2 或 1.0 时给予线性惩罚
    low_danger = max(0.0, 0.3 - body_z)
    high_danger = max(0.0, body_z - 0.9)
    height_penalty = -2.0 * (low_danger + high_danger)

    # ---------- 直立姿态约束（二次惩罚） ----------
    # 身体直立投影 body_up_z = 1 - 2*(quat_x^2 + quat_y^2)，越接近 1 越好
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z  # 0 表示完全直立
    upright_penalty = -2.0 * (upright_error**2)

    # ---------- 力矩效率约束（极小权重，防极端动作） ----------
    action_cost = -0.01 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + height_penalty + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components