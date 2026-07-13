def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next state
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # distance to target (landing pad)
    dist = (x**2 + y**2)**0.5 + 1e-6
    # unit vector pointing toward target
    ux = -x / dist
    uy = -y / dist

    # velocity alignment with target direction (positive when moving closer)
    alignment = vx * ux + vy * uy
    progress = 0.2 * max(0.0, alignment)

    # stability: favor low speed, low angle, low angular velocity
    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward: base contact + bonus for both legs contacting while stable
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

    # joint progress/stability signal
    joint = (stability * progress + 1e-12) ** 0.5

    # small engine penalty
    engine_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total = joint + contact + engine_penalty

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact,
        "engine_penalty": engine_penalty,
    }
    return float(total), components