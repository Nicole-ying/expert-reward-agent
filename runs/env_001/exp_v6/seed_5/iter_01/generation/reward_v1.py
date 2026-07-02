def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty: discourage high speeds, large angle, and angular velocity in the new state
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.01 * abs(next_obs[4]) - 0.005 * abs(next_obs[5])

    # Soft landing proxy: small bonus when close, slow, upright and both supports in contact
    landing_proxy = 0.0
    if (dist_next < 0.3 and speed_next < 0.3 and abs(next_obs[4]) < 0.2 and
        next_obs[6] == 1.0 and next_obs[7] == 1.0):
        landing_proxy = 1.0

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components