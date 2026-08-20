def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Encourage moving toward the pad center.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 2.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: landing_proxy (joint_condition_proxy, geometric mean)
    #   Reward simultaneous satisfaction of all landing conditions:
    #   feet contact, centered position, low velocity, low angle,
    #   low angular velocity.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # bounded proximity factors: 1/(1 + k*|error|)
    proximity_x = 1.0 / (1.0 + 10.0 * abs(next_obs[0]))   # target radius ~0.1
    proximity_y = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))   # pad height ~0
    low_xvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[2]))    # tolerable x speed
    low_yvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[3]))    # tolerable y speed
    low_angle   = 1.0 / (1.0 + 10.0 * abs(next_obs[4]))   # upright angle
    low_angvel  = 1.0 / (1.0 + 2.0 * abs(next_obs[5]))    # angular velocity

    num_factors = 7
    product = (both_contacts * proximity_x * proximity_y *
               low_xvel * low_yvel * low_angle * low_angvel)
    # geometric mean to prevent collapse of a single factor
    landing_proxy_reward = 4.0 * (product ** (1.0 / num_factors))

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Suppress large angles and angular velocities at all times.
    # ------------------------------------------------------------
    angle_penalty = 2.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.5 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: action_penalty (action_efficiency)
    #   Small fuel penalty for any engine use (actions 1,2,3).
    # ------------------------------------------------------------
    action_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total_reward = (progress_reward +
                    landing_proxy_reward +
                    orientation_stability_reward +
                    action_penalty)

    components = {
        "progress_reward": progress_reward,
        "landing_proxy_reward": landing_proxy_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "action_penalty": action_penalty,
    }

    return float(total_reward), components