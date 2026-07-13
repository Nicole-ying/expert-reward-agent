def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation to a target platform.
    Key changes from previous version:
      - settle_proxy: product -> sum of independent factors (denser signal)
      - new velocity_damping: proximity-gated speed penalty to prevent crash
      - reduced distance penalty weight to avoid dominating other signals
    """
    # Previous state (for delta calculations)
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    # Next state (result of action)
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Shared helpers ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5
    speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # proximity gate: ~1 when close to target, decays with distance
    k_prox = 5.0
    proximity_gate = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Component A: progress_to_goal (delta-distance) ---
    # Reward moving toward target, penalize moving away.
    # Positive when distance decreases.
    w_progress = 1.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # --- Component B: distance_penalty (small steady pull toward target) ---
    # Prevents loitering far away; kept small to not dominate.
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # --- Component C: velocity_damping (proximity-gated) ---
    # Penalise high speed, scaled by proximity: strong near target, weak far away.
    # This is the key signal to prevent crash landings.
    w_vel = 1.5
    velocity_damping = -w_vel * proximity_gate * speed

    # --- Component D: settle_reward (sum of independent factors) ---
    # Each factor rewards one aspect of a good landing.
    # Using SUM instead of PRODUCT so partial progress gives partial reward.
    velocity_factor = 1.0 / (1.0 + 3.0 * speed)       # ~1 when slow
    angle_factor = 1.0 / (1.0 + 3.0 * abs(nangle))    # ~1 when upright
    contact_factor = 0.5 * (nleft_contact + nright_contact)  # ~1 when both legs down

    # Gate by proximity: settle bonuses only matter when near the platform
    w_settle = 2.0
    settle_reward = w_settle * proximity_gate * (velocity_factor + angle_factor + contact_factor)

    # --- Component E: orientation_penalty (proximity-gated) ---
    # Penalise tilt and rotation, stronger near target.
    k_gate = 5.0
    orient_gate = 1.0 / (1.0 + k_gate * curr_distance)
    w_orient = 0.5
    w_angvel = 0.15
    orientation_penalty = -w_orient * orient_gate * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # --- Component F: engine_efficiency ---
    # Small penalty for any engine use to encourage fuel efficiency.
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_reward
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_reward": float(settle_reward),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components