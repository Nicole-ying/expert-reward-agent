def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: state_to_improvement — replace absolute approach_quality_reward
    with potential-based shaping: reward = w * (Φ(s') - Φ(s)).
    Φ = proximity * speed_factor * angle_factor, same quality factors as v3.
    Coefficient w=80 matches the new value range (total bounded by ~80).
    """
    # -- Distance to target
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (unchanged)
    w_vel    = 0.001
    w_angle  = 0.001
    w_angvel = 0.0001

    stability_penalty = (
        -w_vel    * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle  * abs(next_obs[4])
        -w_angvel * abs(next_obs[5])
    )

    # -- 3. Landing potential shaping (replaces approach_quality_reward)
    # Quality factors for current state
    proximity_curr    = 1.0 / (1.0 + 5.0 * dist_current)
    speed_curr        = abs(obs[2]) + abs(obs[3])
    speed_factor_curr = 2.0 / (2.0 + speed_curr)
    angle_factor_curr = 1.0 / (1.0 + abs(obs[4]))
    potential_current = proximity_curr * speed_factor_curr * angle_factor_curr

    # Quality factors for next state
    proximity_next    = 1.0 / (1.0 + 5.0 * dist_next)
    speed_next        = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor_next = 2.0 / (2.0 + speed_next)
    angle_factor_next = 1.0 / (1.0 + abs(next_obs[4]))
    potential_next = proximity_next * speed_factor_next * angle_factor_next

    w_potential = 80.0
    landing_shaping_reward = w_potential * (potential_next - potential_current)

    # -- Total reward
    total_reward = progress_reward + stability_penalty + landing_shaping_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward
    }

    return float(total_reward), components