def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current distance to target (from current obs)
    dist_before = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # Next distance to target (from next_obs)
    dist_after = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # 1. Progress signal: reward reduction in distance, penalize increase
    #    Coefficient 5.0 scales the typically small per-step delta (~0.01-0.03)
    #    to be comparable with other reward components
    distance_reward = (dist_before - dist_after) * 5.0

    # 2. Stability constraint: discourage high speed, tilt, and rotation
    stab_weight = 0.01
    stability_penalty = -stab_weight * (
        abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[4]) + abs(next_obs[5])
    )

    # 3. Soft landing proxy: bonus when near target, low speed, upright, both legs contact
    near_target = dist_after < 0.2
    low_speed = abs(next_obs[2]) < 0.2 and abs(next_obs[3]) < 0.2
    stable_angle = abs(next_obs[4]) < 0.1
    low_ang_vel = abs(next_obs[5]) < 0.1
    both_legs = (next_obs[6] == 1.0) and (next_obs[7] == 1.0)

    if near_target and low_speed and stable_angle and low_ang_vel and both_legs:
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components