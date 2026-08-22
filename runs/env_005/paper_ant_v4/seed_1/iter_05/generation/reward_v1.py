def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract relevant observations ----
    body_z = obs[0]
    # quaternion components
    quat_x = obs[2]
    quat_y = obs[3]
    # forward (world x) velocity
    body_x_vel = obs[13]

    # ---- compute health indicators ----
    # target body height (midpoint of healthy range)
    target_height = 0.6
    height_error = body_z - target_height

    # body up vector z-component in world frame (derived from quaternion)
    # body_up = [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx^2 + qy^2)]
    body_up_z = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)

    # tilt error: 0 when perfectly upright, >0 when tilting
    tilt_error = 1.0 - body_up_z

    # combine height and tilt errors into a single posture error
    posture_error = (height_error * height_error + tilt_error * tilt_error) ** 0.5

    # exponential health gate: 1 when posture_error = 0, decays towards 0 as error grows
    k_gate = 5.0
    health_gate = 2.718281828 ** (-k_gate * posture_error)

    # ---- forward progress signal ----
    # only reward moving forward (positive x velocity)  
    forward_speed = body_x_vel if body_x_vel > 0.0 else 0.0
    # bounded mapping to avoid unbounded credit for extremely high speeds
    forward_bounded = forward_speed / (1.0 + forward_speed)

    # ---- total reward ----
    # multiplicative combination: agent must maintain good posture to earn forward progress credit
    w_forward = 8.0
    total_reward = w_forward * forward_bounded * health_gate

    # components dict for logging (total_reward itself is the only additive component)
    components = {
        'progress': total_reward
    }

    return float(total_reward), components