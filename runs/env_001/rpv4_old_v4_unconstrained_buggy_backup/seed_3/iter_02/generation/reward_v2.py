def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: landing-as-global-optimum — direct distance gradient + large sparse contact bonus
    
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

    # --- 1. goal_proximity: direct distance penalty ---
    # Continuous, non-decaying gradient toward (0,0).
    # Using sqrt (Euclidean) gives linear falloff; Manhattan would also work.
    # Weight chosen so that at typical start distance ~8, penalty ≈ -0.4/step,
    # which over ~200-step approach accumulates ~-80, clearly beaten by +100 contact.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_damping: stronger near ground ---
    # Quadratic penalty on speed, gated by proximity to pad.
    # Increased weight (0.05→0.15) to meaningfully discourage high-speed landing.
    w_v = 0.15
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: softened to not dominate ---
    # Reduced angle weight (0.1→0.05) so positive signals aren't drowned.
    # Keeps angular velocity damping to suppress wobble.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus: large sparse reward for successful landing ---
    # Relaxed thresholds (x<0.5, y<0.3) vs. previous (x<0.2, y<0.1).
    # Discrete +100 bonus makes landing the unambiguous global optimum.
    # Early training (cp10%) showed agent CAN find contact — now it will be reinforced.
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