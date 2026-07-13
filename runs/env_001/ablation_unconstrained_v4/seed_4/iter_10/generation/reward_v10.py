def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- helper: distances ---
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # --- progress (convexify for stronger gradient near target) ---
    dist_reduction = max(0.0, prev_dist - next_dist)
    progress = 2.0 * (dist_reduction ** 2)   # squared to amplify when reduction is small

    # --- speed penalty (hinge: penalize only above safe threshold) ---
    v_thresh = 0.5
    speed_penalty = -2.0 * max(0.0, speed - v_thresh)

    # --- angle penalty (hinge, keep upright) ---
    a_thresh = 0.15
    angle_penalty = -5.0 * max(0.0, angle - a_thresh)

    # --- contact quality reward (both legs touch) ---
    contact = next_obs[6] * next_obs[7]  # 0 or 1
    v_target = 0.2
    a_target = 0.1
    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = 20.0 * contact * quality

    # --- total ---
    total_reward = progress + speed_penalty + angle_penalty + contact_reward

    components = {
        'progress': progress,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }
    return float(total_reward), components