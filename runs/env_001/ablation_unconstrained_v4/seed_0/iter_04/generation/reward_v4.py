def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Hybrid reward: progress-driven approach + distance-gated settle + landing bonus.
    
    Key changes from failed settle_delta:
    - Restore absolute settle_reward (distance-gated, only near target)
    - Increase progress_delta weight (3→10) as primary approach driver
    - Reduce distance_penalty (0.5→0.2) to ease negative pressure
    - Reduce velocity_damping (1.5→0.6) and orientation penalty
    - Add large landing_bonus for stable terminal state (self-limiting via episode termination)
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speeds ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (for velocity/orientation damping) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Distance gate for settle reward: active only near target ---
    settle_dist_gate = max(0.0, 1.0 - curr_distance / 2.0)  # 1 at target, 0 beyond dist 2.0

    # ============================================================
    # Component A: progress_delta (primary approach guidance, STRONG)
    # ============================================================
    w_progress = 10.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (gentle baseline pull)
    # ============================================================
    w_dist = 0.2
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated, reduced)
    # ============================================================
    w_vel = 0.6
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: settle_reward (absolute, distance-gated)
    # ============================================================
    # Quality factors: speed, angle, contact
    speed_quality = 1.0 / (1.0 + 5.0 * curr_speed)
    angle_quality = 1.0 / (1.0 + 5.0 * abs(nangle))
    contact_score = 0.5 * (nleft_contact + nright_contact)
    settle_quality = settle_dist_gate * speed_quality * angle_quality * (1.0 + contact_score)
    w_settle = 3.0
    settle_reward = w_settle * settle_quality

    # ============================================================
    # Component E: orientation_penalty (proximity-gated, reduced)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # ============================================================
    # Component G: landing_bonus (strong terminal incentive)
    # ============================================================
    # Only when truly landed: both legs down, stable, near target
    # Episode naturally terminates soon after, preventing farming
    landing_conditions = (
        nleft_contact > 0.5 and nright_contact > 0.5 and
        curr_distance < 0.3 and curr_speed < 0.2 and abs(nangle) < 0.1
    )
    w_landing = 40.0
    landing_bonus = w_landing if landing_conditions else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_reward
        + orientation_penalty
        + engine_penalty
        + landing_bonus
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_reward": float(settle_reward),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
        "landing_bonus": float(landing_bonus),
    }

    return float(total_reward), components