def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v8: Replace punitive vel_damping with positive soft_approach_reward —
        a gradient that pulls the agent toward the pad AND rewards deceleration.
        Boost state-based landing_bonus to make touchdown clearly better than hovering.
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

    # ── 3. Soft approach reward (POSITIVE gradient near pad + low speed) ─
    # Replaces punitive vel_damping: agent earns reward for being close
    # AND slow, creating a funnel: approach → decelerate → contact.
    # Contact-independent so it guides behaviour BEFORE first touchdown.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    w_soft = 2.0
    sigma_dist = 1.5                                    # landing-zone radius
    sigma_speed = 0.5                                   # acceptable approach speed
    soft_approach_reward = (
        w_soft *
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * sigma_dist ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * sigma_speed ** 2))
    )

    # ── 4. State-based landing quality reward ─────────────────────────
    # Per-step reward when legs are in contact with the pad.
    # State-based (not transition-based): no bounce exploit because
    # bouncing = losing contact = losing reward.
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)

    # Quality gates: reward scales with how good the landing state is
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 40.0                                    # boosted from 20 → strong touchdown incentive
    if curr_contacts >= 2.0:
        # Both legs in contact: full landing quality reward
        landing_bonus = w_landing * quality
    elif curr_contacts >= 1.0:
        # Single-leg contact: 20% partial credit for exploration gradient
        landing_bonus = 0.2 * w_landing * quality
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