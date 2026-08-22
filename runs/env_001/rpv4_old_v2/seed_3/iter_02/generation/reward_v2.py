def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (event_based landing bonus)
    #   Reward only the moment both legs first touch the pad.
    #   No continuing reward for staying on the pad.
    # ------------------------------------------------------------
    legs_touching_now = (obs[6] > 0.5) and (obs[7] > 0.5)
    legs_touching_next = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    if (not legs_touching_now) and legs_touching_next:
        safe_contact_reward = 20.0
    else:
        safe_contact_reward = 0.0

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components