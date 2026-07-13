def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    distance = (x**2 + y**2) ** 0.5
    proximity = -1.0 * distance

    # stability 重构为 gate 因子：值域 (0,1]，低速度小角度时接近 1
    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    contact = 0.5 * (left_contact + right_contact)

    total = stability * proximity + contact

    components = {
        "proximity": proximity,
        "stability": stability,
        "contact": contact
    }
    return float(total), components