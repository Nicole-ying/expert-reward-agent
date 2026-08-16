def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4 reward: added descent_incentive — continuous gradient for final descent phase

    Key change from v3: new `descent_incentive` component rewards downward progress
    when the agent is near the landing zone (y < 0.8, |x| < 0.5). This bridges the
    gap between "hovering near target" (where goal_proximity gradient vanishes at
    low y due to squared-distance form) and "committing to touchdown."
    
    Contact_bonus is kept at w=4.0 as a terminal landing confirmation, now reachable
    thanks to the continuous approach gradient provided by descent_incentive.
    
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
    # Encourage moving toward (0,0). Larger weight on vertical axis to avoid hard landings.
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (NEW — continuous descent gradient near landing zone) ---
    # Reward downward progress when near the landing pad.
    # Gates ensure this only fires when the agent is in position to land:
    #   near_ground: linearly decays from 1 at y=0 to 0 at y>=0.8
    #   near_center: linearly decays from 1 at x=0 to 0 at |x|>=0.5
    #   descending:  positive only when moving downward (y_cur > y_nxt)
    # This provides a continuous gradient that peaks during the final descent,
    # naturally tapering to zero once landed (y≈0, no further descent possible).
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)       # active below y=0.8
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descending = max(0.0, y_cur - y_nxt)             # positive when going down
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus) ---
    # Reward when both legs touch and position is near (0,0).
    # Kept at moderate weight — descent_incentive now provides the approach gradient.
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
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