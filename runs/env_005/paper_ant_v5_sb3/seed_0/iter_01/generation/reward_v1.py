def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]          # not used but available
    body_x_vel = next_obs[13]
    # body_y_vel and body_z_vel available but not used in v1

    # 1. Forward progress: reward positive forward velocity
    w_forward = 1.0
    forward = w_forward * body_x_vel    # negative when moving backward (natural penalty)

    # 2. Upright orientation: penalty for deviation from perfectly upright
    # body_up_z = 1 - 2*(quat_x^2 + quat_y^2)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    w_upright = 10.0
    upright_pen = -w_upright * upright_error**2

    # 3. Healthy body height: hinge penalty only when approaching termination bounds
    # safe zone (0.25, 0.95), margins before actual termination at [0.2, 1.0]
    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    total_reward = forward + upright_pen + height_pen

    components = {
        "forward": forward,
        "upright": upright_pen,
        "height": height_pen
    }

    return float(total_reward), components