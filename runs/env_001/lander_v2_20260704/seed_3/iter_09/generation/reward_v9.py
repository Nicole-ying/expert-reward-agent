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

    # 3. Proximity quality: always-active bounded signal for landing-ready configuration.
    #    Rational functions 1/(1+k*x) never collapse to exactly zero,
    #    providing gradient toward low distance, low speed, and upright attitude.
    dist = d_next
    proximity_quality = 2.0 * (1.0 / (1.0 + 2.0 * dist)) * (1.0 / (1.0 + speed)) * (1.0 / (1.0 + 5.0 * angle))

    # 4. Contact bonus: additive reward for foot contact, decoupled from proximity product.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    contact_bonus = 2.0 * contact_avg

    total_reward = approach_reward + stability_penalty + proximity_quality + contact_bonus

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "proximity_quality": proximity_quality,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components