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

    vel_penalty = -0.05 * (vx**2 + vy**2)
    ang_penalty = -0.02 * (ang_vel**2)
    angle_penalty = -0.5 * (angle**2)
    stability = 2.0 * (vel_penalty + ang_penalty + angle_penalty)

    contact = 0.5 * (left_contact + right_contact)

    total = proximity + stability + contact

    components = {
        "proximity": proximity,
        "stability": stability,
        "contact": contact
    }
    return float(total), components