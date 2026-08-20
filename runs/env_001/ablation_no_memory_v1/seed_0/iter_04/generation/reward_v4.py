def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Compute Euclidean distance to target (landing pad center)
    d_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Main learning signal: progress towards the target
    progress = d_curr - d_next
    progress_clipped = max(-0.5, min(0.5, progress))
    progress_reward = 10.0 * progress_clipped

    # Stability penalty: discourage high speed, large angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.05 * abs(vx) - 0.02 * abs(vy) - 0.05 * abs(angle) - 0.01 * abs(angular_vel)

    # Landing quality improvement reward
    # Replaces persistent state-based proxy with improvement-based reward.
    # Only rewards getting better at landing, not occupying a good state.
    landing_zone = 0.3

    if d_next < landing_zone:
        # Quality of current state (before step) — zero if outside landing zone
        if d_curr < landing_zone:
            prox_c = max(0.0, 1.0 - d_curr / landing_zone)
            speed_c = (obs[2]**2 + obs[3]**2)**0.5
            vel_q_c = max(0.0, 1.0 - speed_c / 0.5)
            angle_q_c = max(0.0, 1.0 - abs(obs[4]) / 0.3)
            contact_q_c = 0.5 * (obs[6] + obs[7])
            q_curr = prox_c * (vel_q_c + angle_q_c + 0.5 * contact_q_c)
        else:
            q_curr = 0.0

        # Quality of next state (after step)
        prox_n = max(0.0, 1.0 - d_next / landing_zone)
        speed_n = (next_obs[2]**2 + next_obs[3]**2)**0.5
        vel_q_n = max(0.0, 1.0 - speed_n / 0.5)
        angle_q_n = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
        contact_q_n = 0.5 * (next_obs[6] + next_obs[7])
        q_next = prox_n * (vel_q_n + angle_q_n + 0.5 * contact_q_n)

        quality_improvement = max(0.0, q_next - q_curr)
        landing_quality_reward = 15.0 * quality_improvement
    else:
        landing_quality_reward = 0.0

    total_reward = progress_reward + stability_penalty + landing_quality_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_quality_reward': landing_quality_reward
    }
    return float(total_reward), components