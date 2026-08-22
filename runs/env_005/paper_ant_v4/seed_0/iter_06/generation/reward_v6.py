def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 提取 next_obs 的状态 ---
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    # 速度
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel  = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel   = next_obs[18]

    # --- 健康门控因子（基于 next_obs）---
    # 1. 高度健康：body_z >= 0.5 完全健康，0.5->0.2 线性衰减到 0
    if body_z >= 0.5:
        health_gate = 1.0
    elif body_z <= 0.2:
        health_gate = 0.0
    else:
        health_gate = (body_z - 0.2) / 0.3

    # 2. 直立健康
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    if body_up_z >= 0.5:
        upright_gate = 1.0
    else:
        upright_gate = max(0.0, body_up_z / 0.5)

    combined_gate = health_gate * upright_gate

    # --- 奖励组件 ---
    w_fwd = 1000.0
    forward_reward = w_fwd * body_x_vel * combined_gate

    w_lat = 2.0
    lat_penalty = w_lat * body_y_vel**2

    w_ang = 0.5
    ang_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }

    return float(total_reward), components