def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 7): landing_approach replaced with contact_bonus.
    The environment card identifies contact_success as a mandatory role —
    the only explicit "landing complete" signal. Iter 6's landing_approach
    was a continuous descent potential that never peaked at the landing event,
    providing only 0.974 ep_sum with zero successful landings (0/20).

    contact_bonus = 20 * centering_factor (moderate, between the dominating
    50 of iter 2 and the dead 8 of iter 4-5). No velocity gate — the
    velocity_penalty already enforces soft landing, and iter 3's velocity
    gate (exp(-3*speed²)) caused score collapse from 64→21 by making the
    bonus too hard to trigger.

    Velocity penalty kept at 0.3 (softened in iter 6 from 0.5) to avoid
    the "penalty wall" that prevented the agent from approaching the pad.
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

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Restores the mandatory contact_success role (removed in iter 6).
    # Value 20 is chosen between the dominating 50 (iter 2) and dead 8 (iter 4-5).
    # centering_factor ensures off-pad landings receive proportionally less.
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