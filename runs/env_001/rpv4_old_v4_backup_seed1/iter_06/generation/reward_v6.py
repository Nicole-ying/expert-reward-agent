def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 6): contact_bonus replaced with landing_approach —
    a continuous potential-based reward that provides gradient all the way
    to the surface. Unlike the sparse binary contact_bonus (which oscillated
    between dead and 98% magnitude-share across iters 1-5), this signal:
      - rewards descending when horizontally centered (x≈0)
      - suppresses descent reward when far from center (via exp(-5x²))
      - provides uniform per-step gradient at all altitudes (no dead zones)
      - cannot explode: total per-episode reward is bounded by the potential
        difference between start and goal states.

    Velocity penalty weight reduced 0.5→0.3 to soften the "near-ground
    penalty wall" and allow the new landing_approach signal to effectively
    guide the final descent phase.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.3 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- landing approach: continuous surface-proximity potential ---
    # phi(x, y) = -y * exp(-alpha * x²)
    #   High potential when low (small y) AND centered (small |x|).
    #   The exp(-5x²) centering gate suppresses descent reward when the
    #   agent is far from the pad, preventing it from diving at wrong spots.
    # Reward = phi(next) - phi(prev):
    #   + when descending while centered (both factors improve)
    #   + when centering while low (x-improvement matters more near surface)
    #   - when rising or moving away from center
    prev_h = max(obs[1], 0.0)
    next_h = max(next_obs[1], 0.0)
    alpha = 5.0
    prev_phi = -prev_h * (2.718281828 ** (-alpha * obs[0]**2))
    next_phi = -next_h * (2.718281828 ** (-alpha * next_obs[0]**2))
    landing_approach = 2.0 * (next_phi - prev_phi)

    total_reward = goal_proximity + vel_penalty + upright_penalty + landing_approach

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "landing_approach": landing_approach
    }
    return total_reward, components