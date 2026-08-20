def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: sparse_to_dense — replace binary soft_landing_proxy (0.6% active)
    with continuous approach_quality_reward that gives gradient everywhere.
    Progress delta and stability penalty unchanged from v2.
    """
    # -- Distance to target
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (unchanged from v2)
    w_vel    = 0.001
    w_angle  = 0.001
    w_angvel = 0.0001

    stability_penalty = (
        -w_vel    * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle  * abs(next_obs[4])
        -w_angvel * abs(next_obs[5])
    )

    # -- 3. Continuous approach quality (replaces sparse soft_landing_proxy)
    # Soft bounded factors: all in (0, 1], never collapse to exactly zero
    proximity    = 1.0 / (1.0 + 5.0 * dist_next)
    speed        = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor = 2.0 / (2.0 + speed)
    angle_factor = 1.0 / (1.0 + abs(next_obs[4]))

    approach_quality = proximity * speed_factor * angle_factor
    w_approach = 1.0
    approach_quality_reward = w_approach * approach_quality

    # -- Total reward
    total_reward = progress_reward + stability_penalty + approach_quality_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "approach_quality_reward": approach_quality_reward
    }

    return float(total_reward), components