def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation to a target platform.
    Mandatory roles:
      - progress_to_goal: negative Euclidean distance to target
      - successful_settle: joint condition proxy (close, slow, upright, both legs on platform)
      - engine_efficiency: penalty for any engine thrust
    Conditional role:
      - orientation_penalty: gated by distance, penalises angle and angular velocity
    """
    # Current state (not used much, but available)
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # Next state (result of action)
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Component A: progress_to_goal ---
    # Negative Euclidean distance (high when far, near zero when close)
    distance = (nx ** 2 + ny ** 2) ** 0.5
    w_dist = 1.0
    progress_reward = -w_dist * distance

    # --- Component B: successful_settle (joint condition proxy) ---
    # Each factor ranges [0,1]; product encourages simultaneous satisfaction.
    # proximity: high when close
    k_prox = 5.0
    proximity_factor = 1.0 / (1.0 + k_prox * distance)

    # velocity: high when speed is low
    k_vel = 5.0
    speed_sq = nvx ** 2 + nvy ** 2
    velocity_factor = 1.0 / (1.0 + k_vel * speed_sq)

    # angle: high when upright
    k_angle = 4.0
    angle_factor = 1.0 / (1.0 + k_angle * abs(nangle))

    # contact: encourages both legs in contact
    contact_factor = 0.5 * (nleft_contact + nright_contact)

    w_settle = 10.0
    settle_proxy = w_settle * proximity_factor * velocity_factor * angle_factor * contact_factor

    # --- Component C: orientation_penalty (gated by distance) ---
    # gating weight: 1 when distance=0, decays as distance grows
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * distance)

    w_orient = 1.0
    w_angvel = 0.2
    orientation_penalty = -w_orient * gate * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # --- Component D: engine_efficiency ---
    # Penalise any thrust (action != 0)
    w_engine = 0.1
    engine_penalty = 0.0
    if action != 0:
        engine_penalty = -w_engine

    # Total reward
    total_reward = progress_reward + settle_proxy + orientation_penalty + engine_penalty

    components = {
        "progress_to_goal": float(progress_reward),
        "successful_settle_proxy": float(settle_proxy),
        "orientation_penalty": float(orientation_penalty),
        "engine_efficiency": float(engine_penalty),
    }

    return float(total_reward), components