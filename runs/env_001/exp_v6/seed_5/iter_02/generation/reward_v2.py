def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty: discourage high speeds, large angle, and angular velocity in the new state
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.01 * abs(next_obs[4]) - 0.005 * abs(next_obs[5])

    # Continuous landing proxy: product of bounded factors provides gradient throughout approach.
    # Each factor max(0, 1 - value/threshold) decays linearly from 1→0 as the dimension worsens.
    # This replaces the binary if-condition that had 0.46% trigger rate with a signal that
    # activates as soon as the agent enters the approach zone.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)       # active within 1.0 units of pad
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)      # rewards speed < 1.0
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)  # rewards |angle| < 0.5
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0       # both legs down = 1.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components