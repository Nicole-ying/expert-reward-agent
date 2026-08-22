def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5 reward: L1 — contact_bonus weight 4.0→5.0 to close the 8% gap to target 200.

    All other components unchanged from v4. The descent_incentive (added in v4)
    successfully enabled the agent to discover landing; now we amplify the
    terminal landing reward to push the mean score above 200.

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

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (continuous descent gradient near landing zone) ---
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)
    descending = max(0.0, y_cur - y_nxt)
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus, weight ↑ 4.0→5.0) ---
    w_contact = 5.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)