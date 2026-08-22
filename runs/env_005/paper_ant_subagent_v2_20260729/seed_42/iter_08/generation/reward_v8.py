def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    forward_vel = next_obs[13]   # body x velocity
    lateral_vel = next_obs[14]   # body y velocity
    roll_vel    = next_obs[16]   # body roll velocity
    pitch_vel   = next_obs[17]   # body pitch velocity
    body_z      = next_obs[0]    # body height

    # 计算身体直立度：body_up_z = 1 - 2*(qx^2 + qy^2)
    qx = next_obs[2]
    qy = next_obs[3]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)  # 范围[-1,1]，1为完全直立

    # 直立性门控：当 body_up_z 低于 0.3 时门控归零，线性过渡到 1.0 时门控为 1
    uprightness_gate = max(0.0, (body_up_z - 0.3) / 0.7)

    # 主前进奖励（引入直立性门控）
    forward_reward = 2.0 * forward_vel * uprightness_gate

    # 侧向漂移惩罚 (hinge, 容忍 |lateral_vel| <= 0.2)
    lateral_penalty = -0.5 * max(0.0, abs(lateral_vel) - 0.2)

    # 滚转/俯仰角速度惩罚 (hinge, 容忍 |roll|,|pitch| <= 0.5)
    roll_penalty  = -0.2 * max(0.0, abs(roll_vel) - 0.5)
    pitch_penalty = -0.2 * max(0.0, abs(pitch_vel) - 0.5)

    # 身体高度边界惩罚 (hinge: 安全区域 [0.3, 0.9])
    height_low_penalty  = -1.0 * max(0.0, 0.3 - body_z)
    height_high_penalty = -1.0 * max(0.0, body_z - 0.9)

    total_reward = (forward_reward +
                    lateral_penalty + roll_penalty + pitch_penalty +
                    height_low_penalty + height_high_penalty)

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'roll_penalty': roll_penalty,
        'pitch_penalty': pitch_penalty,
        'height_low_penalty': height_low_penalty,
        'height_high_penalty': height_high_penalty,
    }

    return float(total_reward), components