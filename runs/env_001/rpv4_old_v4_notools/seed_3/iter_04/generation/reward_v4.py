def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward: v2 improvements + turn contact signal from dense state reward
                into event-based progress (leg touch events) and a one-time
                success bonus when both legs touch at low vertical speed.
    
    obs / next_obs layout: [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    """
    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur = obs[6]
    right_cont_cur = obs[7]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement delta on squared distance) ---
    #     increased weights to give stronger pull toward the pad
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity damping (gate by distance, same as before) ---
    w_v = 0.05
    k_gate = 1.0
    dist_nxt = abs(x_nxt) + abs(y_nxt)
    gate = 1.0 / (1.0 + k_gate * dist_nxt)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation penalty ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. landing progress (event-based, replaces dense contact_bonus) ---
    # a) leg touch events – one bonus per leg transition 0→1
    w_leg_event = 5.0
    left_touch  = w_leg_event if (left_cont_nxt > left_cont_cur) else 0.0
    right_touch = w_leg_event if (right_cont_nxt > right_cont_cur) else 0.0

    # b) success event – both legs on pad AND low vertical speed (one‑time)
    w_success = 100.0
    success_cur = (left_cont_cur == 1 and right_cont_cur == 1 and abs(obs[3]) < 0.5)
    success_nxt = (left_cont_nxt == 1 and right_cont_nxt == 1 and abs(vy_nxt) < 0.5)
    success_event = w_success if (success_nxt and not success_cur) else 0.0

    landing_progress = left_touch + right_touch + success_event

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + landing_progress

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'landing_progress': landing_progress   # event‑based now
    }

    return float(total_reward), components