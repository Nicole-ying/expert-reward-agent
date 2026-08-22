def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 提取观测 ---
    body_z = obs[0]
    # 四元数分量
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    # 速度
    body_x_vel = obs[13]   # 世界前向速度
    body_y_vel = obs[14]   # 侧向速度
    # 角速度
    roll_vel  = obs[16]
    pitch_vel = obs[17]
    yaw_vel   = obs[18]

    # --- 健康门控因子 ---
    # 1. 高度健康：当 body_z >= 0.5 时视为完全健康，0.5 -> 0.2 线性衰减至 0
    if body_z >= 0.5:
        health_gate = 1.0
    elif body_z <= 0.2:
        health_gate = 0.0
    else:
        health_gate = (body_z - 0.2) / 0.3   # [0.2,0.5] 线性

    # 2. 直立健康：使用 body_up_z = 1 - 2*(quat_x^2 + quat_y^2)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    # body_up_z 通常在 1(完全直立) 到 -1(倒立) 之间，要求在 0.5 以上才有正向门控
    if body_up_z >= 0.5:
        upright_gate = 1.0
    else:
        # 0.5 以下线性衰减，0 时门控为 0
        upright_gate = max(0.0, body_up_z / 0.5)

    # 组合门控
    combined_gate = health_gate * upright_gate

    # --- 奖励组件 ---
    # 主前向速度奖励，由门控调节
    w_fwd = 1000.0
    forward_reward = w_fwd * body_x_vel * combined_gate

    # 侧向漂移惩罚（平方惩罚柔和，避免绝对值死区）
    w_lat = 2.0
    lat_penalty = w_lat * body_y_vel**2

    # 角速度惩罚（惩罚剧烈的身体旋转）
    w_ang = 0.5
    ang_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # 总奖励
    total_reward = forward_reward - lat_penalty - ang_penalty

    # 组件记录
    components = {
        'forward_reward': forward_reward,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }

    return float(total_reward), components