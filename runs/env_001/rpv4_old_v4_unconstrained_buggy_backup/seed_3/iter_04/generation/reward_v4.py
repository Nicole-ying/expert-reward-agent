def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: one-time landing bonus + direct distance gradient

    Key changes from v3:
    1. goal_proximity: revert from potential-based (delta squared distance)
       back to direct distance penalty — the only approach proven stable (v2).
       Provides consistent gradient toward (0,0) at all positions.
    2. contact_bonus: switch from per-step accumulation (100/step, which
       caused 99.6% magnitude dominance in v2) to a ONE-TIME transition bonus
       (+200 on first pad contact) plus a tiny per-step settling incentive (+2).
       This eliminates the "stay and collect" exploit while retaining the
       landing anchor.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur = obs[6]
    right_cont_cur = obs[7]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty (v2 approach, proven stable) ---
    # Consistent gradient toward (0,0) at every step, regardless of movement.
    # Weight 0.05 means at typical start distance ~8, penalty ≈ -0.4/step;
    # over ~200-step approach accumulates ~-80, clearly beaten by +200 contact.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Stronger damping near ground to prevent high-speed crashes.
    w_v = 0.15
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: ONE-TIME transition bonus + tiny settling incentive ---
    # The core fix: instead of 100/step (which dominated everything in v2
    # and caused training collapse in v3), reward the landing EVENT once,
    # then provide a gentle per-step incentive to stay until body settles.
    #
    # Detect whether agent was already on the pad last step (prevents repeat).
    near_target_cur = (abs(x_cur) < 0.5) and (abs(y_cur) < 0.3)
    both_legs_cur = (left_cont_cur > 0.5) and (right_cont_cur > 0.5)
    was_on_pad = near_target_cur and both_legs_cur

    near_target_nxt = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs_nxt = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    is_on_pad = near_target_nxt and both_legs_nxt

    # Transition: first step entering the pad region with both legs down.
    just_landed = is_on_pad and not was_on_pad

    w_contact_one_time = 200.0   # one-time reward for the landing event
    w_settle_per_step = 2.0      # tiny per-step incentive to maintain contact

    contact_bonus = 0.0
    if just_landed:
        contact_bonus += w_contact_one_time
    if is_on_pad:
        contact_bonus += w_settle_per_step

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components