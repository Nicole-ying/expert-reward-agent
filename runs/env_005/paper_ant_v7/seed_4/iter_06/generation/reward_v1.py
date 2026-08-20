def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]                         # quaternion (not used directly)
    body_x_vel = next_obs[13]                # forward velocity
    body_y_vel = next_obs[14]                # lateral velocity
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]                   # angular velocities

    # ---------- 1. soft health gate: upright factor ----------
    # body_up_z = 1 - 2*(qx^2 + qy^2), ranges from -1 (inverted) to +1 (perfect upright)
    body_up_z = 1.0 - 2.0 * (qx*qx + qy*qy)
    # tilt_error (0 when perfectly upright, up to 2 when inverted)
    tilt_error = 1.0 - body_up_z
    # gate: decays exponentially when tilted, using e^{-lambda * tilt_error}
    lambda_up = 3.0
    upright_gate = 2.718281828 ** (-lambda_up * tilt_error)   # ~1 when upright, ~0 when tilted

    # main progress signal: forward velocity, gated by upright
    gated_forward = body_x_vel * upright_gate

    # ---------- 2. height hinge penalty ----------
    lower_safe = 0.30
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    # scale to keep penalty comparable but lighter than progress reward
    height_penalty_weighted = -10.0 * height_penalty

    # ---------- 3. lateral velocity penalty ----------
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    # ---------- 4. angular velocity penalty ----------
    ang_penalty = -0.02 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # ---------- combine ----------
    w_forward = 2.0
    total_reward = (w_forward * gated_forward
                    + height_penalty_weighted
                    + lateral_penalty
                    + ang_penalty)

    components = {
        "gated_forward": w_forward * gated_forward,
        "height_hinge_penalty": height_penalty_weighted,
        "lateral_penalty": lateral_penalty,
        "angular_velocity_penalty": ang_penalty
    }

    return float(total_reward), components