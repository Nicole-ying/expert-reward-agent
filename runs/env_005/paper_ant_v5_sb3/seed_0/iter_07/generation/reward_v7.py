def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_x_vel = next_obs[13]
    quat_x = next_obs[2]
    quat_y = next_obs[3]

    # 直立因子（body_up_z ∈ [-1, 1]，1 为全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 前向奖励（仅当身体朝上时）
    forward = 1.0 * body_x_vel * upright_factor

    # 直立偏差 hinge 惩罚（只在倾斜过大时激活）
    threshold = 0.5
    w_hinge = 1.5
    upright_pen = -w_hinge * max(0.0, threshold - body_up_z)

    total_reward = forward + upright_pen

    components = {
        "forward": forward,
        "upright_pen": upright_pen
    }
    return float(total_reward), components