def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward: rebalanced — reduced contact_bonus dominance, stronger goal_proximity

    Key change from v2: contact_bonus weight reduced 10→4 (was 92.5% magnitude share),
    goal_proximity weights doubled (w_x 1→2, w_y 2→4) to restore continuous guidance.
    This balances the sparse landing bonus against the continuous approach signals.
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, weights doubled) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    # This lets the agent fly fast at high altitude but forces deceleration for landing.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy, weight reduced) ---
    # reward when both legs touch and position is near (0,0)
    # weight reduced from 10.0 to 4.0 to prevent dominance over continuous signals
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)