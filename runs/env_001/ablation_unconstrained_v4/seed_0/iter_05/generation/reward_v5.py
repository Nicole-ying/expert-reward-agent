def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes exploit of settle_reward and sparse landing_bonus cliff:
    - Removes standalone settle_reward (was 100% active, 35.7% magnitude share)
    - Removes binary landing_bonus (2.8% active, 40-point cliff)
    - Replaces both with continuous landing_reward: strict bounded product
      (distance < 0.3, speed < 0.25, angle < 0.15 rad), weight 6
    - Increases progress_delta weight 10→12 as primary approach driver
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (for damping terms) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (PRIMARY approach driver)
    # ============================================================
    w_progress = 12.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (gentle baseline pull)
    # ============================================================
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated)
    # ============================================================
    w_vel = 0.5
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: orientation_penalty (proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component E: engine_penalty
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # ============================================================
    # Component F: landing_reward (continuous bounded product)
    # Replaces exploit-prone settle_reward and sparse landing_bonus.
    # Each factor uses max(0, 1 - value/threshold) for smooth gradient.
    # Strict thresholds ensure activation only in final landing phase.
    # ============================================================
    dist_factor = max(0.0, 1.0 - curr_distance / 0.3)
    speed_factor = max(0.0, 1.0 - curr_speed / 0.25)
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.15)
    contact_bonus = 0.5 * (nleft_contact + nright_contact)
    landing_quality = dist_factor * speed_factor * angle_factor * (1.0 + contact_bonus)
    w_landing = 6.0
    landing_reward = w_landing * landing_quality

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + orientation_penalty
        + engine_penalty
        + landing_reward
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
        "landing_reward": float(landing_reward),
    }

    return float(total_reward), components