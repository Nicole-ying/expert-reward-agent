def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 4): goal_proximity switched from linear delta-distance
    to inverse-distance potential — provides much stronger per-step gradients,
    especially near the ground where fine deceleration control matters.
    contact_bonus magnitude reduced from 50→8 and velocity gating removed
    (gating created a chicken-and-egg problem in iter 3, dropping score from
    64→21). The lower bonus magnitude prevents it from drowning out the new
    dense progress signal.
    """
    # --- goal proximity: inverse-distance potential for dense gradients ---
    # phi(d) = 1/(d + epsilon)  ->  reward = phi(next) - phi(prev)
    # This form amplifies gradients near the target (where soft landing
    # requires precise control) while still providing directional signal
    # at moderate distances.
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    epsilon = 0.5  # prevents singularity at d=0, controls gradient sharpness
    prev_potential = 1.0 / (prev_dist + epsilon)
    next_potential = 1.0 / (next_dist + epsilon)
    goal_proximity = 3.0 * (next_potential - prev_potential)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate reward for dual-leg contact ---
    # Velocity gating removed (iter 3 regression evidence).
    # Magnitude reduced 50→8 to prevent drowning dense progress signal.
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