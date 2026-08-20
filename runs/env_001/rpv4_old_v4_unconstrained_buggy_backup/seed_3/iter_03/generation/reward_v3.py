def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: potential-based approach shaping + proven sparse contact bonus
    
    Key change from v2: goal_proximity switched from direct distance penalty
    (always negative, weak gradient) to potential-based improvement reward
    (positive when approaching target, negative when retreating).
    This gives the agent meaningful gradient during the ~88% of steps before contact,
    while the proven contact_bonus remains as the terminal anchor.
    
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

    # --- 1. goal_proximity: potential-based improvement reward ---
    # Rewards reduction in squared distance to target (0,0).
    # delta > 0 means approaching; delta < 0 means retreating.
    # Squared distance gives naturally stronger gradient when far away
    # and gentler when close, encouraging rapid initial approach then precision.
    # Weights moderate to avoid dominating the contact_bonus anchor.
    w_x = 0.3
    w_y = 0.6
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive = approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive = approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping: stronger near ground ---
    # Quadratic penalty on speed, gated by proximity to pad.
    # Slightly increased weight (0.15→0.20) to counterbalance the stronger
    # approach gradient and discourage high-speed dives.
    w_v = 0.20
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep the lander upright.
    # Weights unchanged from v2 — these worked adequately.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus: large sparse reward for successful landing ---
    # Unchanged from v2 — relaxed thresholds + large per-step bonus.
    # This is the proven anchor that makes landing the global optimum.
    # Per-step nature rewards staying on the pad until body settles.
    w_contact = 100.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components