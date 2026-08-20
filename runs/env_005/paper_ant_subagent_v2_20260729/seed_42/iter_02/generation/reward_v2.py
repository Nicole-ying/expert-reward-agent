def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract necessary observations
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_roll_velocity = obs[16]
    body_pitch_velocity = obs[17]

    # Compute body up vector z-component (uprightness)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(-1.0, min(1.0, body_up_z))  # numerical safety

    # Health gate: height
    height_low = 0.2
    height_low_margin = 0.15   # 0.2 + 0.15 = 0.35
    height_high = 0.95
    height_high_margin = 0.05  # 0.95 - 0.05 = 0.90

    # Height gate: linear decay to 0 when approaching boundaries
    if body_z < height_low:
        height_gate = 0.0
    elif body_z < height_low + height_low_margin:
        height_gate = (body_z - height_low) / height_low_margin
    elif body_z > height_high:
        height_gate = 0.0
    elif body_z > height_high - height_high_margin:
        height_gate = (height_high - body_z) / height_high_margin
    else:
        height_gate = 1.0

    # Uprightness gate: tightened thresholds
    upright_threshold = 0.4   # below this -> gate 0
    upright_full = 0.8        # above this -> gate 1
    if body_up_z < upright_threshold:
        upright_gate = 0.0
    elif body_up_z < upright_full:
        upright_gate = (body_up_z - upright_threshold) / (upright_full - upright_threshold)
    else:
        upright_gate = 1.0

    gate = height_gate * upright_gate

    # Forward progress (main signal)
    w_forward = 1.0
    forward_progress = w_forward * max(0.0, body_x_velocity) * gate

    # Lateral drift penalty (hinge)
    w_lat = 0.5
    lateral_drift_penalty = -w_lat * max(0.0, abs(body_y_velocity) - 0.3)

    # Roll/pitch stability hinge
    w_rp = 0.2
    ang_vel_margin = 1.0  # rad/s
    roll_penalty = max(0.0, abs(body_roll_velocity) - ang_vel_margin)
    pitch_penalty = max(0.0, abs(body_pitch_velocity) - ang_vel_margin)
    roll_pitch_penalty = -w_rp * (roll_penalty + pitch_penalty)

    total_reward = forward_progress + lateral_drift_penalty + roll_pitch_penalty

    components = {
        'forward_progress': forward_progress,
        'lateral_drift_hinge': lateral_drift_penalty,
        'roll_pitch_hinge': roll_pitch_penalty
    }

    return float(total_reward), components