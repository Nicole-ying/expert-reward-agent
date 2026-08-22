def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the result state (next_obs)
    body_z = next_obs[0]
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # Quaternion-based body uprightness: z-component of body's local up in world frame
    body_up_z = qw * qw - qx * qx - qy * qy + qz * qz
    body_up_z = max(-1.0, min(1.0, body_up_z))

    # Height health: 1 at center (0.6), 0 at boundaries (0.25, 0.95)
    height_health = 1.0 - abs(body_z - 0.6) / 0.35
    height_health = max(0.0, min(1.0, height_health))

    # Stability gate: only reward forward motion when upright and at healthy height
    stability_gate = max(0.0, body_up_z) * height_health

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 10.0
    w_orient = 0.5

    # 1. Main learning signal: forward velocity reward, gated by stability
    forward_reward = w_fwd * body_x_vel * stability_gate

    # 2. Stability constraint (penalty for dangerous/inefficient deviations)
    lateral_penalty = w_lat * abs(body_y_vel)
    angular_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)
    height_penalty = w_h * (
        max(0.0, 0.25 - body_z) + max(0.0, body_z - 0.95)
    )
    stability_penalty = -(lateral_penalty + angular_penalty + height_penalty)

    # 3. Orientation reward: reward staying upright to prevent collapse
    orientation_reward = w_orient * body_up_z

    total_reward = forward_reward + stability_penalty + orientation_reward

    components = {
        "forward_velocity_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "orientation_reward": orientation_reward
    }

    return float(total_reward), components