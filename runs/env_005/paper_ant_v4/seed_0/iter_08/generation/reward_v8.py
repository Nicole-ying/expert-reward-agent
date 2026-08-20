def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # 前进奖励：平方强化高速
    fwd_speed = max(0.0, body_x_vel)
    forward_reward = 15.0 * (fwd_speed ** 2)

    # 软约束：高度惩罚
    height_low = max(0.0, 0.3 - body_z)
    height_high = max(0.0, body_z - 0.9)
    height_penalty = 20.0 * (height_low + height_high)

    # 直立惩罚：连续惩罚，偏离完全直立即产生代价
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_penalty = 5.0 * ((1.0 - body_up_z) ** 2)

    # 横向速度惩罚
    lat_penalty = 2.0 * (body_y_vel ** 2)

    # 角速度惩罚
    ang_penalty = 0.5 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - height_penalty - upright_penalty - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'height_penalty': -height_penalty,
        'upright_penalty': -upright_penalty,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }
    return float(total_reward), components