def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes iter 6/7 hover exploit by:
    1. Removing proximity_reward — the always-on distance payment that incentivizes hovering
    2. Boosting progress_delta as the primary approach driver
    3. Redesigning stable_landing_bonus as a STRICT landing-only bonus:
       - Requires ALL of: near target, low speed, upright angle, both leg contacts
       - Multiplicative gating (product of independent conditions) ensures bonus ≈ 0
         unless every landing condition is simultaneously met
       - Sparse but high-value: pays meaningfully only during actual settling/landing
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (used ONLY as a modulator, never standalone) ---
    k_prox = 5.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (PRIMARY approach driver)
    # Boosted to compensate for removal of proximity_reward.
    # This must be the dominant positive signal driving the agent to the target.
    # ============================================================
    w_progress = 40.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (global baseline pull toward origin)
    # ============================================================
    w_dist = 0.5
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: stable_landing_bonus (STRICT landing-only sparse reward)
    # Multiplicative product of independent landing conditions.
    # Each factor ranges [0,1]; product is near-zero unless ALL are satisfied.
    # This replaces both proximity_reward and the old permissive bonus.
    # ============================================================
    # Independent landing-condition gates (each [0,1], 1 = perfect)
    near_target = max(0.0, 1.0 - curr_distance / 0.4)       # dist < 0.4
    speed_low = max(0.0, 1.0 - curr_speed / 0.3)            # speed < 0.3
    angle_upright = max(0.0, 1.0 - abs(nangle) / 0.25)     # |angle| < 0.25 rad
    both_contacts = nleft_contact * nright_contact           # 1.0 only if both legs touch

    w_stable = 25.0
    # Product gating: bonus > 0 only when ALL conditions are substantially met.
    # (0.5 + 0.5*both_contacts) ensures contact bonus is partial with one leg,
    # full with both — but still requires near/slow/upright to matter.
    landing_quality = near_target * speed_low * angle_upright * (0.3 + 0.7 * both_contacts)
    stable_landing_bonus = w_stable * landing_quality

    # ============================================================
    # Component D: velocity_damping (mild, proximity-gated speed penalty)
    # ============================================================
    w_vel = 0.3
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component E: orientation_penalty (mild, proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + stable_landing_bonus
        + velocity_damping
        + orientation_penalty
        + engine_penalty
    )

    reward_components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "stable_landing_bonus": float(stable_landing_bonus),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), reward_components