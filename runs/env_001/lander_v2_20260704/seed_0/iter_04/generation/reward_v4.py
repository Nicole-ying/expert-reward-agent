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

    # 3. Landing quality as improvement (potential-based shaping)
    def quality_potential(o):
        d = (o[0] ** 2 + o[1] ** 2) ** 0.5
        proximity_gate = max(0.0, 1.0 - d / 0.5)
        speed_quality = max(0.0, 1.0 - (abs(o[2]) + abs(o[3])) / 0.5)
        angle_quality = max(0.0, 1.0 - abs(o[4]) / 0.3)
        angvel_quality = max(0.0, 1.0 - abs(o[5]) / 0.3)
        contact_score = (o[6] + o[7]) / 2.0
        return 2.0 * proximity_gate * (
            speed_quality + angle_quality + angvel_quality + contact_score
        ) / 4.0

    quality_before = quality_potential(obs)
    quality_after = quality_potential(next_obs)

    # Improvement-based: only reward change in landing quality
    landing_quality = (quality_after - quality_before) * 10.0

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components