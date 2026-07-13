def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    prev_lc = obs[6]
    prev_rc = obs[7]

    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5

    proximity_reward = -0.1 * (x**2 + y**2)

    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    contact_transition = max(0.0, (next_lc * next_rc) - (prev_lc * prev_rc))
    contact_bonus = 10.0 * contact_transition

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total_reward), components