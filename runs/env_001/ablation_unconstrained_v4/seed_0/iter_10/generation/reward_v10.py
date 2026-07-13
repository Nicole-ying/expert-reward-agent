def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes iter 8 crash-by-overspeed by:
    1. Reducing progress_delta from 40→12 — guide, not dominate
    2. Replacing strict multiplicative landing_bonus (0.6% active) with 
       proximity-gated settle_reward using additive condition factors
    3. Distance penalty reduced 0.5→0.2
    4. Speed penalty strengthened near target: w_speed = 0.5 + 3.0*prox
       to directly incentivize deceleration during final approach
    Based on iter 2 settle_reward success pattern.
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (modulator only) ---
    k_prox = 5.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (MODEST approach guide)
    # Reduced from 40→12. Must not dominate over landing incentives.
    # ============================================================
    w_progress = 12.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (mild baseline pull)
    # Reduced from 0.5→0.2 to prevent penalty stack overwhelming
    # ============================================================
    w_dist = 0.2
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: settle_reward (proximity-gated additive landing reward)
    # Replaces the strict multiplicative product (0.6% active).
    # Each landing condition independently contributes reward when near target.
    # near_factor is the master proximity gate; within it, slow/upright/contact
    # each add their own bounded contribution.
    # ============================================================
    near_factor = max(0.0, 1.0 - curr_distance / 0.5)        # within 0.5 units
    slow_factor = max(0.0, 1.0 - curr_speed / 0.4)           # speed < 0.4
    upright_factor = max(0.0, 1.0 - abs(nangle) / 0.3)       # |angle| < 0.3 rad
    contact_sum = nleft_contact + nright_contact              # 0, 1, or 2

    w_settle = 20.0
    # Additive within proximity gate: each factor contributes independently
    # so partial landing progress is rewarded, not just perfect completion.
    settle_quality = near_factor * (
        0.4 * slow_factor +
        0.3 * upright_factor +
        0.8 * min(contact_sum / 2.0, 1.0)
    )
    settle_reward = w_settle * settle_quality

    # ============================================================
    # Component D: speed_penalty (proximity-strengthened)
    # Mild far from target (0.5), strong near target (up to 3.5).
    # Directly incentivizes deceleration during final approach.
    # ============================================================
    w_speed_base = 0.5
    w_speed_prox = 3.0
    w_speed = w_speed_base + w_speed_prox * curr_prox
    speed_penalty = -w_speed * curr_speed

    # ============================================================
    # Component E: orientation_penalty (mild regularizer)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + settle_reward
        + speed_penalty
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "settle_reward": float(settle_reward),
        "speed_penalty": float(speed_penalty),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components