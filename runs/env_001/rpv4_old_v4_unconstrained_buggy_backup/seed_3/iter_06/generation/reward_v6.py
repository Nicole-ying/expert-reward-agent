def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: potential-based progress reward + per-step contact bonus (iter_2 skeleton)

    Core diagnosis from v5 failure:
    - v5's always-negative distance penalty (goal_proximity=-36.24, 82.2% share)
      created a pure-penalty landscape. Agent learned "hover far away, do nothing"
      because every step toward the target was punished.
    - contact_bonus (25/step) was too small to overcome the penalty gradient;
      agent found landing at cp60% but then abandoned it.
    - Only iter_2's per-step contact_bonus (100) ever successfully activated
      landing behavior. All attempts to "fix" the 99.6% dominance killed it.

    Changes from v5:
    1. goal_proximity: switched from distance penalty (-w*dist) to potential-based
       progress reward: w_pot * (dist_cur - dist_nxt). Positive when approaching,
       negative when retreating. This ensures every step toward target is net-positive
       regardless of contact, restoring exploration gradient.
    2. contact_bonus: per-step weight raised 25→80 — close to iter_2's proven 100,
       but slightly moderated since progress reward now also provides positive signal.
    3. velocity_penalty: slightly reduced 0.4→0.3 — progress reward already encodes
       a soft speed incentive (approaching requires some velocity).

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

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Unlike distance penalty (always negative), this makes every approach step
    # net-beneficial, restoring the exploration gradient toward the target.
    # Weight 2.0 means approaching 0.1 units/step yields +0.2 reward.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized so progress reward can drive initial approach.
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged — these have been stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (iter_2 proven): rewards every step with both legs on pad
    # near target. Creates natural curriculum from first touch → stable settle.
    # Weight 80: close to iter_2's 100 (which worked), moderated because progress
    # reward now provides positive approach gradient.
    w_contact = 80.0
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