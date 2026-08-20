def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 9): contact_bonus (sparse binary) replaced by
    soft_landing_envelope — a continuous multiplicative reward that peaks
    at the ideal landing state (h=0, x=0, v=0). Unlike the sparse bonus
    which oscillated between dead (0%) and dominating (98% magnitude share)
    across 7 of 8 iterations, the envelope provides gradient everywhere:
      - exp(-h/0.3): sharp height decay — negligible above h=1, ramps near ground
      - exp(-5x²): centering — same gate proven effective in iters 2/7/8
      - exp(-(vx²+vy²)): slows the agent as it approaches the pad

    Velocity penalty gate softened from exp(-h/1.0) to exp(-h/1.5):
    the iter 8 gate was too sharp, creating a "penalty cliff" near the
    surface that the agent learned to avoid entirely. exp(-h/1.5) provides
    a smoother transition while keeping the penalty concentrated near ground.

    The multiplicative envelope ensures all landing criteria must be
    simultaneously satisfied for high reward — partial satisfaction
    (e.g., centered but fast) yields proportionally less.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 at ground, decays smoothly with height (k=1.5)
    # At h=1.5: gate=0.368, h=1.0: gate=0.513, h=0.5: gate=0.717, h=0: gate=1.0
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.5)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- soft landing envelope: continuous reward peaking at ideal landing ---
    # Three multiplicative factors, each in (0, 1], product peaks at 1.0
    # at the perfect landing state (h=0, x=0, vx=vy=0).
    # Peak value = 3.0 per step at ideal state.
    # At h=0.5, x=0.1, v²=0.1: ~3.0 * 0.189 * 0.951 * 0.905 ≈ 0.49
    # At h=1.0, x=0.2, v²=0.3: ~3.0 * 0.036 * 0.819 * 0.741 ≈ 0.065
    # At h=2.0, x=0.5, v²=1.0: ~3.0 * 0.0013 * 0.287 * 0.368 ≈ 0.0004
    h = max(next_obs[1], 0.0)
    h_factor = 2.718281828 ** (-h / 0.3)
    x_factor = 2.718281828 ** (-5.0 * next_obs[0]**2)
    v_factor = 2.718281828 ** (-(next_obs[2]**2 + next_obs[3]**2))
    soft_landing_envelope = 3.0 * h_factor * x_factor * v_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + soft_landing_envelope

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "soft_landing_envelope": soft_landing_envelope
    }
    return float(total_reward), components