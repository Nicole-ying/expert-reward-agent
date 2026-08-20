def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_x_vel = next_obs[13]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_z = next_obs[0]

    # 直立因子（body_up_z ∈ [-1, 1]，1 为全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 前向奖励（仅当身体朝上时）
    forward = 1.0 * body_x_vel * upright_factor

    # 身体高度铰链惩罚：在高度接近 0.2 前发出警告
    height_threshold = 0.3
    w_height_pen = 1.0
    height_pen = -w_height_pen * max(0.0, height_threshold - body_z)

    # 直立惩罚：当身体不够直立时施加，防止因倾斜导致跌倒
    w_upright_pen = 0.5
    upright_threshold = 0.8
    upright_pen = -w_upright_pen * max(0.0, upright_threshold - body_up_z)

    total_reward = forward + height_pen + upright_pen

    components = {
        "forward": forward,
        "height_pen": height_pen,
        "upright_pen": upright_pen
    }
    return float(total_reward), components