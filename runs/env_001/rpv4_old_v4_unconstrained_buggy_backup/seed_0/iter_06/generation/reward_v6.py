def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v6: Replace punitive vel_damping with positive soft_approach_reward
        (reward low speed near pad instead of penalising speed);
        restore landing_bonus to moderate weight + moderate quality gates
        (between iter_4's loose/exploitable and iter_5's tight/dead);
        differentiate full landing vs partial contact, anti-bounce via first-contact gate.
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

    # ── 3. Soft approach reward (POSITIVE reward for low speed near pad) ─
    # Replaces punitive vel_damping: agent earns reward for decelerating,
    # creating a positive gradient instead of a cost to be "tolerated".
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin
    speed = (vx ** 2 + vy ** 2) ** 0.5
    w_soft = 0.4
    sigma_speed = 0.6                                    # moderate: rewards meaningful slowing
    soft_approach_reward = (
        w_soft * gate *
        2.718281828 ** (- (speed ** 2) / (2.0 * sigma_speed ** 2))
    )

    # ── 4. Landing contact bonus (transition-based, anti-bounce) ─────
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)

    # Moderate quality gates: between iter_4 (1.5/0.5 → loose, exploitable)
    # and iter_5 (0.8/0.3 → tight, dead).  Graded enough to guide learning
    # but suppress high-speed bounce re-trigger.
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.4 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 45.0                                     # between 30 (dead) and 80 (exploit)
    # Anti-bounce gate: only reward FIRST contact transition (prev_contacts == 0).
    # Once any leg has touched, subsequent re-contacts earn no bonus.
    # This blocks the iter_4 bounce-exploit while preserving the initial touchdown signal.
    if prev_contacts == 0.0 and new_contacts >= 2.0:
        landing_bonus = w_landing * quality
    elif prev_contacts == 0.0 and new_contacts >= 1.0:
        landing_bonus = 0.3 * w_landing * quality        # partial credit for first single-leg touch
    else:
        landing_bonus = 0.0

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
        soft_approach_reward +
        landing_bonus +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_approach_reward": soft_approach_reward,
        "landing_bonus": landing_bonus,
        "action_penalty": action_penalty
    }
    return float(total_reward), components