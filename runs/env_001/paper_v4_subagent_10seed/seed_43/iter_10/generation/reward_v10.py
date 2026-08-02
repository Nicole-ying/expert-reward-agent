def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]

    # Distance to target
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5
    delta = dist_old - dist_new

    # Health gate: based on body angle and speed
    angle_healthy = 1.0 / (1.0 + 2.0 * angle**2)
    speed = abs(vx) + abs(vy)
    speed_healthy = 1.0 / (1.0 + 0.5 * speed)
    gate = angle_healthy * speed_healthy

    # Progress reward
    w_progress = 3.0
    progress_reward = w_progress * max(0.0, delta) * gate

    # Contact success bonus
    contact_reward = 0.0
    if left_contact == 1.0 and right_contact == 1.0:
        x_thresh = 0.5
        y_thresh = 0.5
        v_thresh = 1.0
        angle_thresh = 0.5

        closeness = max(0.0, 1.0 - abs(x)/x_thresh) * max(0.0, 1.0 - y/y_thresh)
        stability = max(0.0, 1.0 - (abs(vx) + abs(vy))/v_thresh) * max(0.0, 1.0 - abs(angle)/angle_thresh)
        w_contact = 6.0   # ← increased from 5.0
        contact_reward = w_contact * closeness * stability

    # Angular velocity penalty (hinge)
    angvel_limit = 0.5
    w_angvel = 0.5
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_limit)

    total = progress_reward + contact_reward + angvel_penalty

    components = {
        'progress': progress_reward,
        'contact_success': contact_reward,
        'angvel_penalty': angvel_penalty
    }
    return float(total), components