def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 5): goal_proximity reverted from inverse-distance delta
    back to linear distance delta. Inverse-distance potential (iter 4) caused
    catastrophic training collapse: negligible gradients at moderate distances
    meant the agent received almost no approach signal, learning to hover
    instead. Linear delta provides uniform per-step gradient at all distances,
    which was proven to work in iter 2 (score 63.72, 17/20 terminated).
    Weight raised to 4.0 (from iter 2's 2.0) to compensate for the smaller
    contact_bonus (8 vs 50) and ensure net continuous signals are positive.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate reward for dual-leg contact ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 8.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components