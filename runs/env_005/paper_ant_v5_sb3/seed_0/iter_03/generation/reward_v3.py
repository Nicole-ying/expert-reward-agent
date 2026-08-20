def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]

    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    upright_gate = max(0.0, 1.0 - 2.0 * upright_error)
    
    w_forward = 1.0
    forward = w_forward * body_x_vel * upright_gate

    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    total_reward = forward + height_pen

    components = {
        "forward": forward,
        "height": height_pen
    }
    return float(total_reward), components