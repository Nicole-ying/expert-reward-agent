def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 8): velocity penalty gate sharpened from exp(-h/2.0)
    to exp(-h/1.0) and weight raised from 0.3 to 0.5. This concentrates
    the soft-landing penalty near the ground (h < 1.0) while giving the
    agent more freedom at altitude. The sharper near-ground gradient forces
    deceleration in the final descent phase, bridging the gap between
    "approaching the pad" and "actually landing."

    Evidence: iter 7 had only 2/20 landings despite active goal_proximity.
    The velocity penalty at -0.3 with the wide gate (exp(-h/2)) was too
    diffuse to create a clear "slow down now" signal near the surface.
    Iter 2 (best, 17/20) used w=0.5, and the sharper gate retains altitude
    freedom while making the near-ground braking signal even stronger than
    iter 2's configuration.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: sharply height-gated ---
    # gate = 1.0 at ground, decays rapidly with height (k=1.0 vs old k=2.0)
    # At h=1.0: gate=0.368 (old: 0.607) — weaker, more freedom at altitude
    # At h=0.3: gate=0.741 (old: 0.861) — comparable
    # At h=0.0: gate=1.000 — full penalty strength
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Value 20 kept from iter 7 — large enough to incentivize landing,
    # small enough not to dominate when continuous signals are healthy.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 20.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components