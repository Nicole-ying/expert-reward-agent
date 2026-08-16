def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6 reward: L2 — widened descent_incentive gate + higher weight to end the
    contact_bonus lottery.

    Problem (v4→v5): contact_bonus alternates between dead (0% active) and dominant
    (86-92% mag share). When the agent accidentally discovers landing, it learns;
    when it doesn't, training fails completely. The descent_incentive bridge (added
    in v4) is too weak at w=2.0 and gate y<0.8 to independently guide the agent.

    Changes from v5:
      - descent_incentive: near_ground gate widened 0.8→1.5 (active from higher),
        weight increased 2.0→6.0. This provides ~3-5x stronger continuous descent
        gradient, reducing dependence on sparse contact_bonus.
      - contact_bonus: weight reverted 5.0→4.0 (iter_4 proven value).

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

    # --- 4. descent_incentive (L2 — widened gate + stronger weight) ---
    # Gate widened from 0.8 to 1.5: agent now gets descent gradient from higher
    # altitude, making the signal less dependent on already being near the pad.
    # Weight increased 2.0→6.0: the per-step gradient should now be comparable
    # to goal_proximity (~8 ep_sum), strong enough to guide landing independently.
    w_desc = 6.0
    near_ground = max(0.0, 1.0 - y_nxt / 1.5)       # active below y=1.5 (was 0.8)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descending = max(0.0, y_cur - y_nxt)             # positive when going down
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event, weight reverted to 4.0) ---
    w_contact = 4.0
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