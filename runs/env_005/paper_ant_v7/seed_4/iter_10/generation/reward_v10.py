def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    body_z = obs[0]          # torso height
    quat_x = obs[2]          # quaternion x component
    quat_y = obs[3]          # quaternion y component
    body_x_vel = obs[13]     # forward velocity
    body_y_vel = obs[14]     # lateral velocity
    roll_vel  = obs[16]      # body angular velocities
    pitch_vel = obs[17]
    yaw_vel   = obs[18]

    # Compute uprightness
    uprightness = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    uprightness = max(-1.0, min(1.0, uprightness))

    # Soft health gate – height
    height_gate = (body_z - 0.25) / (0.35 - 0.25)
    height_gate = max(0.0, min(1.0, height_gate))

    # Soft health gate – uprightness
    upright_gate = (uprightness - 0.5) / (0.9 - 0.5)
    upright_gate = max(0.0, min(1.0, upright_gate))

    # Combined gate
    forward_gate = min(height_gate, upright_gate)

    # Main signal: gated forward velocity
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * forward_gate

    # Stabilisation: suppress lateral drift (unchanged)
    w_lat = 0.1
    lateral_penalty = -w_lat * (body_y_vel**2)

    # Stabilisation: damp excessive body rotation (CHANGED to hinged penalty)
    angular_thresh = 0.3
    roll_penalty  = max(0.0, abs(roll_vel)  - angular_thresh) ** 2
    pitch_penalty = max(0.0, abs(pitch_vel) - angular_thresh) ** 2
    yaw_penalty   = max(0.0, abs(yaw_vel)   - angular_thresh) ** 2
    angular_penalty = -0.1 * (roll_penalty + pitch_penalty + yaw_penalty)

    total_reward = forward_reward + lateral_penalty + angular_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
    }
    return float(total_reward), components