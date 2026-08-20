def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraints: light penalty on speed, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * speed - 0.01 * angle - 0.005 * ang_vel

    # 3. Contact-gated landing quality: zero when no foot contact, rewards good landing posture.
    #    Rational decay 1/(1+kx) preserves gradient farther from optimum than exponentials.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    dist = d_next
    soft_landing_proxy = 3.0 * contact_avg / ((1.0 + 2.0 * dist) * (1.0 + speed) * (1.0 + 5.0 * angle))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components