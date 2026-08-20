def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: Dense-dominated reward. Progress toward (0,0) is the main driver (~200-300/ep).
    Sparse contact/landing bonuses are mild supplements (~10-20/ep).
    No per-step drain. Speed penalty near ground for soft landing.
    """
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- 1. progress: THE main signal. Strong potential-based gradient toward (0,0) ---
    # Telescoping sum = 30 * (dist_start - dist_end). From dist~10 to 0: ~300 reward.
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 30.0 * (dist_c - dist_n)

    # --- 2. descent_reward: extra push to go DOWN when horizontally centered ---
    # No speed gate — speed is handled separately. Just reward descending.
    x_centered = max(0.0, 1.0 - abs(x_n) / 2.0)
    descending = max(0.0, y_c - y_n)
    descent_reward = 8.0 * x_centered * descending

    # --- 3. speed_cost: penalize high speed near ground (squared, so severe only when fast) ---
    # This is the soft-landing shaper. Strong above speed~1.0, negligible below ~0.3.
    near_ground = max(0.0, 1.0 - abs(y_n) / 1.5)
    speed_sq = vx_n ** 2 + vy_n ** 2
    speed_cost = -4.0 * near_ground * speed_sq

    # --- 4. orient_cost: mild attitude regularizer ---
    orient_cost = -0.1 * abs(angle_n) - 0.02 * (angvel_n ** 2)

    # --- 5. contact_reward: mild supplement for leg contact ---
    contact_reward = 1.5 * (lc_n + rc_n)

    # --- 6. landing_bonus: small bonus for both legs on pad near origin ---
    near_origin = max(0.0, 1.0 - (x_n ** 2 + y_n ** 2) ** 0.5 / 0.4)
    landing_bonus = 3.0 * lc_n * rc_n * near_origin

    # --- 7. fuel_cost: symbolic only, don't discourage necessary engine use ---
    if action == 2:
        fuel_cost = -0.01
    elif action in (1, 3):
        fuel_cost = -0.005
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + descent_reward
        + speed_cost
        + orient_cost
        + contact_reward
        + landing_bonus
        + fuel_cost
    )

    components = {
        'progress': progress,
        'descent_reward': descent_reward,
        'speed_cost': speed_cost,
        'orient_cost': orient_cost,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components