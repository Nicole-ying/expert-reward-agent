def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- continuous proximity signal (absolute distance penalty + incremental progress)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # global pressure to reduce distance (cannot be evaded by local oscillation)
    proximity_pressure = -0.4 * dist_next

    # incremental progress – strong shaping towards the target
    progress = 2.0 * (dist_prev - dist_next)

    # --- simple velocity penalty (no altitude gating, applies everywhere)
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # --- orientation penalty (unchanged rationale, small tweak)
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- contact state reward (not incremental → no exploit by lifting/placing repeatedly)
    # each leg contact gives a small constant positive reward
    contact_state = 0.3 * (float(next_obs[6]) + float(next_obs[7]))

    # --- combine
    total_reward = proximity_pressure + progress + velocity_penalty + orientation_penalty + contact_state

    components = {
        "proximity_pressure": proximity_pressure,
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_state": contact_state
    }

    return float(total_reward), components