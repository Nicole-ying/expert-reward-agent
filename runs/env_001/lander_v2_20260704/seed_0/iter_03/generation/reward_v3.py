def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current distance to target (from current obs)
    dist_before = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # Next distance to target (from next_obs)
    dist_after = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Progress signal: reward reduction in distance, penalize increase
    distance_reward = (dist_before - dist_after) * 5.0

    # 2. Stability constraint: discourage high speed, tilt, and rotation
    stab_weight = 0.01
    stability_penalty = -stab_weight * (
        abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[4]) + abs(next_obs[5])
    )

    # 3. Continuous landing quality (replaces sparse binary soft_landing_proxy)
    # Proximity gate: activates gradually within 0.5 distance of target
    proximity_gate = max(0.0, 1.0 - dist_after / 0.5)

    # Quality factors, each in [0, 1]
    speed_quality = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)
    angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    angvel_quality = max(0.0, 1.0 - abs(next_obs[5]) / 0.3)
    contact_score = (next_obs[6] + next_obs[7]) / 2.0

    # Landing quality: proximity-gated average of quality factors, scaled
    landing_quality = 2.0 * proximity_gate * (
        speed_quality + angle_quality + angvel_quality + contact_score
    ) / 4.0

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components