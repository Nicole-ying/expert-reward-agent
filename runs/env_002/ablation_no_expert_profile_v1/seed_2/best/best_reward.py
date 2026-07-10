def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Primary progress: reward forward horizontal velocity (only positive values)
    forward_vel = next_obs[2]
    forward_reward = max(forward_vel, 0.0)

    # Stability health penalties
    hull_angle = next_obs[0]          # body tilt from vertical
    hull_ang_vel = next_obs[1]        # body angular velocity
    vertical_vel = next_obs[3]        # vertical speed (bouncing)

    stability_cost = (
        hull_angle ** 2 +
        hull_ang_vel ** 2 +
        vertical_vel ** 2
    )

    # Component weights (v1)
    w_forward = 1.0
    w_stability = 0.1

    total_reward = w_forward * forward_reward - w_stability * stability_cost

    components = {
        'forward_reward': w_forward * forward_reward,
        'stability_penalty': -w_stability * stability_cost
    }

    return float(total_reward), components