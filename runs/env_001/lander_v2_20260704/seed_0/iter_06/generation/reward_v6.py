def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next distance to target
    dist_before = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_after = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Progress signal: reward reduction in distance, penalize increase
    distance_reward = (dist_before - dist_after) * 5.0

    # 2. Stability constraint: discourage high speed, tilt, and rotation
    stab_weight = 0.01
    stability_penalty = -stab_weight * (
        abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[4]) + abs(next_obs[5])
    )

    # 3. Landing quality as potential-based improvement (state_to_improvement)
    # --- Potential for current state (before action) ---
    proximity_before = max(0.0, 1.0 - dist_before / 0.5)
    contact_raw_before = (obs[6] + obs[7]) / 2.0
    contact_factor_before = 0.3 + 0.7 * contact_raw_before
    speed_factor_before = max(0.0, 1.0 - (abs(obs[2]) + abs(obs[3])) / 0.5)
    angle_factor_before = max(0.0, 1.0 - abs(obs[4]) / 0.3)
    angvel_factor_before = max(0.0, 1.0 - abs(obs[5]) / 0.3)
    pose_quality_before = (speed_factor_before + angle_factor_before + angvel_factor_before) / 3.0
    potential_before = proximity_before * contact_factor_before * pose_quality_before

    # --- Potential for next state (after action) ---
    proximity_after = max(0.0, 1.0 - dist_after / 0.5)
    contact_raw_after = (next_obs[6] + next_obs[7]) / 2.0
    contact_factor_after = 0.3 + 0.7 * contact_raw_after
    speed_factor_after = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)
    angle_factor_after = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    angvel_factor_after = max(0.0, 1.0 - abs(next_obs[5]) / 0.3)
    pose_quality_after = (speed_factor_after + angle_factor_after + angvel_factor_after) / 3.0
    potential_after = proximity_after * contact_factor_after * pose_quality_after

    # Landing improvement: reward increase in potential
    landing_quality = (potential_after - potential_before) * 5.0

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components