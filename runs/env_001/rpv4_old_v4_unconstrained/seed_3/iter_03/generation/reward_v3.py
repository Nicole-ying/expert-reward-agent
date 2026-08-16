def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: one-shot contact bonus (transition-based) + stronger shaping signals

    Core fix: contact_bonus was per-step with w=25, accumulating to 92% of reward.
    Now it fires ONCE on the transition from "no contact" → "contact", with a
    velocity gate that penalises high-speed touchdowns. The continuous shaping
    signals (landing_approach, goal_proximity) are strengthened to fill the gradient
    that contact_bonus previously monopolised.
    """

    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur, right_cont_cur = obs[6], obs[7]
    left_cont_nxt, right_cont_nxt = next_obs[6], next_obs[7]

    # --- helpers ---
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2

    # ================================================================
    # 1. goal_proximity — distance penalty + approach incentive
    #    w_approach doubled (0.4 → 0.8) to give stronger gradient
    # ================================================================
    w_dist = 0.08
    proximity_penalty = -w_dist * dist_nxt

    w_approach = 0.8
    delta_dist = dist_cur - dist_nxt          # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — penalise speed, stronger near the pad
    #    w_v increased 0.15→0.20, k_gate 2.0→3.0 for sharper gating
    # ================================================================
    w_v = 0.20
    k_gate = 3.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)    # ~1 when close, ~0 when far
    velocity_penalty = -w_v * gate * speed_sq

    # ================================================================
    # 3. orientation_penalty — unchanged weight, effective
    # ================================================================
    w_ang = 0.03
    w_angvel = 0.005
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ================================================================
    # 4. landing_approach — weight doubled (0.3 → 0.6),
    #    temperature parameters slightly widened for broader basin
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.30)   # was 0.25
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.40)   # was 0.35
    slow     = 2.718281828 ** (-speed_sq / 1.0)      # was 0.8
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.35) # was 0.3
    landing_approach = 0.6 * near_y * near_x * slow * upright

    # ================================================================
    # 5. contact_bonus — ONE-TIME transition bonus (not per-step!)
    #    Detects the moment both legs first touch down.
    #    Velocity gate ensures high-speed impacts get less credit.
    #    w_contact = 8.0 (was 25.0 per-step → now ~8 once per episode).
    # ================================================================
    w_contact = 8.0
    had_contact = left_cont_cur * right_cont_cur       # 0 or 1, were we already touching?
    has_contact = left_cont_nxt * right_cont_nxt       # 0 or 1, are we touching now?
    new_contact = has_contact * (1.0 - had_contact)    # 1 ONLY on the transition step

    spatial_gate = 2.718281828 ** (-(x_nxt ** 2 + y_nxt ** 2) / 0.5)
    soft_gate    = 2.718281828 ** (-speed_sq / 0.5)    # high speed → bonus shrinks
    contact_bonus = w_contact * new_contact * spatial_gate * soft_gate

    # ================================================================
    # 6. fuel_efficiency — unchanged, tiny regularisation
    # ================================================================
    w_fuel = 0.003
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components