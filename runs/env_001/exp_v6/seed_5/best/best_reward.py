def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty with distance-gating.
    # Gate = 1/(1+5*dist): ~1.0 at pad, ~0.17 at dist=1.0, ~0.05 at dist=4.0.
    # This lets the agent move freely when far away, and only tightens
    # speed/angle/angvel constraints as it approaches the landing zone.
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    dist_gate = 1.0 / (1.0 + 5.0 * dist_next)
    stability_penalty = dist_gate * (
        -0.01 * speed_next
        - 0.01 * abs(next_obs[4])
        - 0.005 * abs(next_obs[5])
    )

    # Continuous landing proxy: product of bounded factors provides gradient
    # throughout approach. Each factor max(0, 1 - value/threshold) decays
    # linearly from 1→0 as the dimension worsens.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'dist_gate': dist_gate,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components