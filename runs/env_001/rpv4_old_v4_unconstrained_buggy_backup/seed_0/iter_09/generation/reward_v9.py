def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v9: Back to iter_4's proven framework (transition landing_bonus + progress + vel_damping).
        Remove soft_approach_reward entirely — it caused hover exploit in both iter_2 and iter_8.
        landing_bonus: first-contact gate (prev_contacts==0) kills bounce exploit;
        quality gates at intermediate tightness (sigma_dist=1.0, sigma_speed=0.5) —
        discoverable but bounce-unprofitable.
        Boost vel_damping (w=0.4) to provide meaningful deceleration gradient.
    """
    # ── Extract signals ──────────────────────────────────────────────
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # ── 1. Progress reward (potential-based: bounded total, no hover exploit) ─
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # >0 when approaching
    w_progress = 12.0
    progress_reward = w_progress * progress

    # ── 2. Orientation penalty ───────────────────────────────────────
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ── 3. Landing contact bonus (transition-based, first-contact-only) ─
    # Count leg contacts before and after this step
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    # Quality gates: between iter_4 (1.5/0.5 — loose, bounce-exploitable)
    # and iter_5 (0.8/0.3 — tight, undiscoverable).
    # These values: rough landing earns ~8-15, clean landing ~50-60, bounce ~0-1.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 60.0
    # First-contact-only gate: bonus fires ONLY when no leg was touching before.
    # This kills the iter_4 bounce exploit — once contact is made, no more transition bonuses.
    # prev_contacts == 0 ensures the agent can't bounce-and-re-trigger.
    if prev_contacts == 0.0 and new_contacts >= 2.0:
        # Both legs touch simultaneously for the first time: full bonus
        landing_bonus = w_landing * quality
    elif prev_contacts == 0.0 and new_contacts >= 1.0:
        # Single-leg first contact: 30% partial credit for exploration gradient
        landing_bonus = 0.3 * w_landing * quality
    else:
        landing_bonus = 0.0

    # ── 4. Directional velocity damping (punitive, near-pad) ─────────
    # Boosted from w=0.2 (iter_4, negligible) to w=0.4 — provides meaningful
    # deceleration gradient as soft_approach replacement.
    # Vertical deadzone (0.6) allows controlled descent without penalty.
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin

    horizontal_penalty = abs(vx)                         # full penalty on lateral drift
    vertical_penalty = max(0.0, abs(vy) - 0.6)           # deadzone: allow moderate descent

    w_vel_damp = 0.4                                     # up from 0.2 → meaningful decel gradient
    vel_damping = -w_vel_damp * gate * (horizontal_penalty + vertical_penalty)

    # ── 5. Action efficiency ─────────────────────────────────────────
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