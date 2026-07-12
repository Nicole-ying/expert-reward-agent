def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键信号
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]

    # 姿态健康因子：用指数衰减将四元数虚部平方和映射到 (0,1]
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 前进奖励（基础量 × 姿态门控）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 高度硬约束（仅在越出安全范围时激活，作为后备保护）
    height_exceed = max(0.0, 0.2 - body_z) + max(0.0, body_z - 1.0)
    height_penalty = -10.0 * height_exceed

    # 新增：动作幅度惩罚（控制代价）
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    total_reward = forward_reward + lateral_penalty + height_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components