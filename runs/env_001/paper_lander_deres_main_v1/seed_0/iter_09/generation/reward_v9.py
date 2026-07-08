def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Skeleton change: replace approach_quality product with descent_incentive +
    landing_settle to break the hovering local optimum.

    Old skeleton (approach_quality = prox * speed * angle product) created a
    local maximum at "hover near (0,0) with low speed and upright attitude" —
    no requirement to actually land.  Contact_reward (iter 8) was exploited
    via leg-dipping (64% contact rate, score dropped 139→120).

    New skeleton rewards the *action* of controlled descent (not the state of
    being near) and provides a settlement bonus gated behind both-legs contact.
    """

    # ----- Unpack observations -------------------------------------------------
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # ----- Component 1: Progress towards target (0, 0) -------------------------
    dist_current = (x ** 2.0 + y ** 2.0) ** 0.5
    dist_next = (next_x ** 2.0 + next_y ** 2.0) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # ----- Component 2: Descent incentive (anti-hover) -------------------------
    # Rewards controlled downward movement when horizontally aligned with target.
    # Hovering (y_vel ≈ 0) gives zero — directly breaks the local optimum.
    horiz_prox = 1.0 / (1.0 + 5.0 * abs(next_x))
    speed = (next_x_vel ** 2.0 + next_y_vel ** 2.0) ** 0.5
    speed_safety = 1.0 / (1.0 + speed)  # dampens high-speed descent
    descent_incentive = 0.3 * horiz_prox * max(0.0, -next_y_vel) * speed_safety

    # ----- Component 3: Landing settle (terminal-quality proxy) ----------------
    # Rewards being settled with BOTH legs in contact, low speed, and upright.
    # Uses min(left, right) so single-leg contact gives zero — prevents dipping.
    both_legs = min(next_left, next_right)  # 1.0 only if both legs contact
    land_speed_factor = 1.0 / (1.0 + speed)
    land_angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    landing_settle = 0.15 * both_legs * land_speed_factor * land_angle_factor

    # ----- Component 4: Attitude penalty (gentle uprighting) -------------------
    attitude_penalty = -0.01 * abs(next_angle)

    # ----- Total ----------------------------------------------------------------
    total_reward = progress + descent_incentive + landing_settle + attitude_penalty

    components = {
        "progress_reward": progress,
        "descent_incentive": descent_incentive,
        "landing_settle": landing_settle,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components