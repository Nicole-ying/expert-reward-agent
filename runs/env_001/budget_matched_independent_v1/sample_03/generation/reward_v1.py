def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (center pad)
    dist = (obs[0]**2 + obs[1]**2)**0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5

    # 1. primary learning signal: approach improvement (improvement_delta)
    progress = 1.0 * (dist - next_dist)

    # 2. stability constraint: body angle near zero (linear penalty)
    angle_penalty = -0.1 * abs(next_obs[4])

    # 3. safety constraint: low velocities (quadratic penalty on both x and y speed)
    speed_penalty = -0.1 * (next_obs[2]**2 + next_obs[3]**2)

    # 4. soft landing proxy: both legs in contact (continuous product)
    contact_reward = 0.2 * (next_obs[6] * next_obs[7])

    total_reward = progress + angle_penalty + speed_penalty + contact_reward
    components = {
        "progress": progress,
        "angle_penalty": angle_penalty,
        "speed_penalty": speed_penalty,
        "contact": contact_reward
    }
    return float(total_reward), components