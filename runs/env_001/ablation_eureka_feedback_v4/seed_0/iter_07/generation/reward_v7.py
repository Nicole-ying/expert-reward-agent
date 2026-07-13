def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # current position
    x0 = obs[0]
    y0 = obs[1]
    # next position
    x1 = next_obs[0]
    y1 = next_obs[1]

    dist0 = (x0**2 + y0**2) ** 0.5
    dist1 = (x1**2 + y1**2) ** 0.5
    delta = dist0 - dist1  # >0 when approaching pad
    progress = 5.0 * max(0.0, delta)

    # next state
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward: base contact + bonus for both legs contacting while stable
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

    joint = (stability * progress + 1e-12) ** 0.5
    total = joint + contact

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact,
    }
    return float(total), components