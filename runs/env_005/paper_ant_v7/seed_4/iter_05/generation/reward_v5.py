def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from observation (current obs, not next_obs)
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]
    body_roll_vel = obs[16]
    body_pitch_vel = obs[17]
    body_yaw_vel = obs[18]

    # ----- uprightness (projection of body z-axis onto world z-axis) -----
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))  # numerical safety clamp

    # ----- core learning signal: forward velocity modulated by uprightness -----
    # Scale up drastically to escape the low‑speed local optimum.
    forward_reward = 10.0 * body_x_vel * body_up_z

    # ----- height survival gate (continuous, bounded [0, 1]) -----
    z_low_safe = 0.4
    z_low_danger = 0.3
    z_high_safe = 0.8
    z_high_danger = 0.9

    if body_z >= z_low_danger and body_z <= z_low_safe:
        height_gate = (body_z - z_low_danger) / (z_low_safe - z_low_danger)
    elif body_z >= z_low_safe and body_z <= z_high_safe:
        height_gate = 1.0
    elif body_z >= z_high_safe and body_z <= z_high_danger:
        height_gate = (z_high_danger - body_z) / (z_high_danger - z_high_safe)
    else:
        height_gate = 0.0

    # Apply gate to forward reward — bad height suppresses all forward credit
    gated_forward = forward_reward * height_gate

    # ----- lateral stability penalty (hinge on |body_y_velocity|) -----
    lateral_threshold = 0.15
    lateral_error = max(0.0, abs(body_y_vel) - lateral_threshold)
    w_lateral = 0.1
    lateral_penalty = -w_lateral * lateral_error

    # ----- rotation stability penalty (hinge on angular velocities) -----
    rotation_threshold = 0.5
    roll_error = max(0.0, abs(body_roll_vel) - rotation_threshold)
    pitch_error = max(0.0, abs(body_pitch_vel) - rotation_threshold)
    yaw_error = max(0.0, abs(body_yaw_vel) - rotation_threshold)
    w_rotation = 0.05
    rotation_penalty = -w_rotation * (roll_error + pitch_error + yaw_error)

    # ----- action penalty for energy efficiency (unchanged) -----
    w_action = 0.03
    torque_penalty = -w_action * sum(action**2) / 8.0

    total_reward = gated_forward + lateral_penalty + rotation_penalty + torque_penalty

    components = {
        "forward_upright_gated": gated_forward,
        "height_gate_value": height_gate,
        "lateral_penalty": lateral_penalty,
        "rotation_penalty": rotation_penalty,
        "torque_penalty": torque_penalty
    }

    return float(total_reward), components