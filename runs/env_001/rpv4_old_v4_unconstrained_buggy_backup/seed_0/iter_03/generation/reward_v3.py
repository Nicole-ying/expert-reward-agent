def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v3: Replace dominant soft_landing_proxy (state-value reward → hovering exploit)
        with contact-transition landing bonus (event-based);
        boost progress weight; add action efficiency penalty.
    """
    # ── Extract signals ──────────────────────────────────────────────
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # ── 1. Progress reward (improvement_delta on Euclidean distance) ─
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # >0 when approaching
    w_progress = 3.0                                     # boosted from 1.0
    progress_reward = w_progress * progress

    # ── 2. Orientation penalty (quadratic, unchanged) ────────────────
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ── 3. Landing contact bonus (transition-based, NOT state-based) ─
    # Count how many legs were touching BEFORE vs AFTER this step
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    # Landing quality: only reward contact when conditions are good
    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.5 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 60.0
    landing_bonus = w_landing * new_contacts * quality

    # ── 4. Near-pad velocity damping (gate from iter_2, kept) ────────
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin
    w_vel_damp = 0.3
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    # ── 5. Action efficiency (mild engine-use penalty) ───────────────
    if action == 0:                                      # no engine
        action_penalty = 0.0
    elif action == 1 or action == 3:                     # orientation engines
        action_penalty = -0.05
    else:                                                # action == 2, main engine
        action_penalty = -0.2

    # ── Assemble ─────────────────────────────────────────────────────
    total_reward = (
        progress_reward +
        orientation_penalty +
        landing_bonus +
        vel_damping +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
        "vel_damping": vel_damping,
        "action_penalty": action_penalty
    }
    return float(total_reward), components