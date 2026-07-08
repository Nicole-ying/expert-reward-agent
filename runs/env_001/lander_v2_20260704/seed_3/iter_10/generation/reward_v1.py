def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Unpack previous obs
    x_prev = obs[0]
    y_prev = obs[1]

    # Distances to target (origin)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5

    # 1. Progress delta – main learning signal (only reward forward progress)
    delta = dist_prev - dist_next
    progress_reward = max(0.0, delta) * 1.0

    # 2. Proximity attraction – smooth attraction to the target area
    sigma = 0.5
    proximity_reward = (2.718281828 ** (-dist_next / sigma)) * 0.5

    # 3. Stability penalty – gentle suppression of high speed, tilt, and spin
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -(
        w_vel * (abs(vx) + abs(vy)) +
        w_angle * abs(angle) +
        w_angvel * abs(angvel)
    )

    # 4. Contact bonus – only rewarded when already near the landing pad
    contact_bonus = 0.0
    if dist_next < 0.3:
        contact_bonus = 0.2 * (left_contact + right_contact)

    total_reward = progress_reward + proximity_reward + stability_penalty + contact_bonus

    components = {
        'progress_reward': progress_reward,
        'proximity_reward': proximity_reward,
        'stability_penalty': stability_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components