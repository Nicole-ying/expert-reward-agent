def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: direct distance penalty + approach bonus + soft-gated contact + landing shaping

    Core idea: "being far from target always costs something" — no more lazy hovering.
    A smooth reward bridge connects "near the pad" → "actually landing".
    """

    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- helpers ---
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2

    # ================================================================
    # 1. goal_proximity — direct penalty + approach incentive
    #    "You pay rent for being away. You earn credit for getting closer."
    # ================================================================
    w_dist = 0.08                     # per-unit-distance penalty
    proximity_penalty = -w_dist * dist_nxt

    w_approach = 0.4                  # bonus per unit of distance reduced
    delta_dist = dist_cur - dist_nxt  # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — penalise speed, stronger near the pad
    # ================================================================
    w_v = 0.15
    k_gate = 2.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 when close, ~0 when far
    velocity_penalty = -w_v * gate * speed_sq

    # ================================================================
    # 3. orientation_penalty — reduced weight (was 0.1, now 0.03)
    #    Goal: stay upright but don't let this dominate proximity
    # ================================================================
    w_ang = 0.03
    w_angvel = 0.005
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ================================================================
    # 4. landing_approach — continuous shaping bridge
    #    Rewards being in a "ready to land" configuration near the pad:
    #    low altitude, centered, slow, upright.
    #    Uses exponential decay — smooth, always > 0, no hard cliffs.
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)    # decays as you go higher
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)    # decays with lateral offset
    slow     = 2.718281828 ** (-speed_sq / 0.8)        # decays with speed
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.3)  # decays with tilt
    landing_approach = 0.3 * near_y * near_x * slow * upright

    # ================================================================
    # 5. contact_bonus — soft-gated, no longer a needle in a haystack
    #    Uses exponential spatial gate instead of hard |x|<0.2 & |y|<0.1.
    #    Contact anywhere gives *some* reward; contact near (0,0) gives full.
    # ================================================================
    w_contact = 25.0
    both_legs = left_cont_nxt * right_cont_nxt          # 0.0 or 1.0
    spatial_gate = 2.718281828 ** (-(x_nxt ** 2 + y_nxt ** 2) / 0.5)
    contact_bonus = w_contact * both_legs * spatial_gate

    # ================================================================
    # 6. fuel_efficiency — tiny nudge toward engine-off when possible
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