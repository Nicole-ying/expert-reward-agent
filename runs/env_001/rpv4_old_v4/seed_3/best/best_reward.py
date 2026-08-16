def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7 reward: L2 — potential-based descent_incentive replaces linear-gate formulation.

    Problem (v6): descent_incentive uses `near_ground * descending` where both terms
    are linear. As y→0, the possible descent per step shrinks (physics), causing the
    gradient to weaken at the most critical moment — the final few cm before touchdown.
    The agent hovers near the pad but cannot bridge the last gap to landing.

    Change:
      - descent_incentive: replaced linear `(1-y/1.5) * (y_cur-y_nxt)` with
        potential-based shaping. Potential φ(y) = 1/(y+0.1). Reward = φ(y_nxt) - φ(y_cur).
        This creates a gradient that naturally INTENSIFIES as y→0, compensating for
        the shrinking descent step size.
      - contact_bonus: weight reduced 4.0→2.0 (supporting change). When contact_bonus
        fires at cp80% (1169 ep_sum), it creates a destabilizing reward spike that
        causes catastrophic forgetting of landing behavior. Lower weight reduces the
        spike while keeping it as a meaningful confirmation signal.

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

    # --- 4. descent_incentive (L2 — potential-based shaping) ---
    # Potential φ(y) = 1/(y+eps): increases as y decreases, with derivative
    # dφ/dy = -1/(y+eps)^2 that grows strongly near y=0. This creates a
    # natural funnel: the same descent step yields MORE reward when closer
    # to the ground, exactly where the linear formulation used to weaken.
    #
    # Gated by near_center so agent only gets descent credit when aligned
    # with the landing pad.
    w_desc = 1.5
    eps = 0.1
    potential_cur = 1.0 / (y_cur + eps)
    potential_nxt = 1.0 / (y_nxt + eps)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descent_incentive = w_desc * near_center * max(0.0, potential_nxt - potential_cur)

    # --- 5. contact_bonus (sparse landing confirmation, reduced weight) ---
    # Weight reduced 4.0→2.0 to prevent the reward spike (observed up to 1169
    # ep_sum at cp80%) from destabilizing training and causing catastrophic
    # forgetting of landing behavior.
    w_contact = 2.0
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