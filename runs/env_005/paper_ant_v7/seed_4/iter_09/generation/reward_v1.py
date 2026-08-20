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

    # Compute uprightness (z-component of body up vector)
    # Assuming unit quaternion and near‑upright the w component ≈ 1.
    uprightness = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    uprightness = max(-1.0, min(1.0, uprightness))

    # Soft health gate – height component
    # Gate opens linearly from 0 at 0.25 m to 1 at 0.35 m.
    height_gate = (body_z - 0.25) / (0.35 - 0.25)
    height_gate = max(0.0, min(1.0, height_gate))

    # Soft health gate – uprightness component
    # Gate opens linearly from 0 at uprightness 0.5 to 1 at 0.9.
    upright_gate = (uprightness - 0.5) / (0.9 - 0.5)
    upright_gate = max(0.0, min(1.0, upright_gate))

    # Combined gate: minimum ensures both health conditions are partially satisfied
    forward_gate = min(height_gate, upright_gate)

    # Main learning signal: gated forward velocity
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * forward_gate

    # Stabilisation: suppress lateral drift
    w_lat = 0.1
    lateral_penalty = -w_lat * (body_y_vel**2)

    # Stabilisation: damp excessive body rotation
    w_ang = 0.05
    angular_penalty = -w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward + lateral_penalty + angular_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
    }
    return float(total_reward), components