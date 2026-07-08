def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander trajectory optimization.

    Reverts to the best-performing skeleton (progress + approach_quality +
    attitude_penalty) and adds a small continuous contact incentive to
    encourage actual touchdown instead of hovering above the target.
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

    # ----- Component 2: Approach quality (continuous shaping) -------------------
    # Three bounded factors [0,1] multiplied: proximity, low speed, upright.
    # Product form ensures all conditions matter — but also means hovering
    # near the target with low speed and good attitude yields reward without
    # landing.  The new contact_reward (component 4) fills that gap.
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2.0 + next_y_vel ** 2.0) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor

    w_approach = 0.15
    approach_quality_reward = w_approach * approach_quality

    # ----- Component 3: Attitude penalty (gentle uprighting force) --------------
    attitude_penalty = -0.01 * abs(next_angle)

    # ----- Component 4: Contact incentive (NEW — the single modification) --------
    # Small continuous reward per leg in contact.  Provides gradient toward
    # touchdown that the approach_quality product alone lacks.  Max ~0.04 when
    # both legs touch, modest vs approach_quality (~0.094 active mean).
    contact_reward = 0.02 * (next_left + next_right)

    # ----- Total ----------------------------------------------------------------
    total_reward = progress + approach_quality_reward + attitude_penalty + contact_reward

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components