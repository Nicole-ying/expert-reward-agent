def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: Eliminate per-step proximity exploitation.

    Core principles:
    - NO per-step location bonus — only delta-distance progress shapes navigation
    - Landing bonus is the LARGEST positive signal, making landing the optimal choice
    - Contact reward encourages ground interaction for discoverability
    - Velocity damping only near pad, for soft landing guidance
    - Action 0 (glide) is neutral — doing nothing forever earns nothing
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    x_prev = obs[0]
    y_prev = obs[1]

    # ── 1. Progress: delta-distance (THE navigation signal) ──
    # Positive when approaching pad, negative when retreating.
    # No per-step reward for just *being* near the pad.
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 1.5 * (dist_prev - dist_curr)

    # ── 2. Orientation penalty (moderate) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad ──
    # Only penalizes excess speed when close to pad
    speed = (vx ** 2 + vy ** 2) ** 0.5
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: any foot on ground is progress ──
    # Per-step reward encourages the agent to descend and try touching
    contact_sum = left_contact + right_contact
    contact_reward = 0.5 * contact_sum

    # ── 5. Landing success bonus: LARGE sparse reward ──
    # This MUST be the most attractive positive signal
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow
    landing_bonus = 50.0 if is_good_landing else 0.0

    # ── 6. Action efficiency ──
    # Glide (action 0) is neutral — no free lunch for doing nothing
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components