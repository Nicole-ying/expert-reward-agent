def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Splits iter 2's exploit-prone settle_reward into two-layer design:
    - proximity_reward: smooth distance-only signal (100% active, no cliff)
    - stable_landing_bonus: proximity-gated stability bonus (soft factors, extra credit)
    Prevents both the hover exploit (iter 2) and the sparse cliff (iter 5).
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (continuous, always active) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (primary approach driver)
    # ============================================================
    w_progress = 15.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (global baseline pull toward origin)
    # ============================================================
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: proximity_reward (smooth, always-active approach incentive)
    # Replaces iter 2's settle_reward pure-proximity base.
    # 100% active, no threshold cliff: provides continuous gradient toward target.
    # ============================================================
    w_prox = 3.0
    proximity_reward = w_prox * curr_prox

    # ============================================================
    # Component D: stable_landing_bonus (proximity-gated stability extra)
    # Only meaningful near target; soft factors avoid hard-threshold cliff.
    # Multiplies proximity so the bonus scales with closeness.
    # ============================================================
    speed_ok = max(0.0, 1.0 - curr_speed / 0.5)
    angle_ok = max(0.0, 1.0 - abs(nangle) / 0.2)
    contact_factor = 0.5 * (nleft_contact + nright_contact)
    w_stable = 4.0
    stable_landing_bonus = w_stable * curr_prox * speed_ok * angle_ok * (1.0 + contact_factor)

    # ============================================================
    # Component E: velocity_damping (global mild speed penalty, prox-gated)
    # ============================================================
    w_vel = 0.3
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component F: orientation_penalty (proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component G: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + proximity_reward
        + stable_landing_bonus
        + velocity_damping
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "proximity_reward": float(proximity_reward),
        "stable_landing_bonus": float(stable_landing_bonus),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components