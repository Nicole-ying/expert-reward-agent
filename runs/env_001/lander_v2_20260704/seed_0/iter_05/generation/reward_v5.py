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

    # 3. Landing quality with contact as soft-multiplier
    # Proximity gate: activates gradually within 0.5 distance of target
    proximity_gate = max(0.0, 1.0 - dist_after / 0.5)

    # Pose quality factors: speed, angle, angular velocity (each in [0, 1])
    speed_quality = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)
    angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    angvel_quality = max(0.0, 1.0 - abs(next_obs[5]) / 0.3)

    # Contact factor: soft floor at 0.3, scales to 1.0 with both legs down
    contact_raw = (next_obs[6] + next_obs[7]) / 2.0
    contact_factor = 0.3 + 0.7 * contact_raw

    # Pose quality without contact (average of three factors)
    pose_quality = (speed_quality + angle_quality + angvel_quality) / 3.0

    # Landing quality: proximity-gated, contact-multiplied pose quality
    landing_quality = 5.0 * proximity_gate * contact_factor * pose_quality

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components