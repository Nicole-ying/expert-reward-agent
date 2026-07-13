def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- weights & thresholds -----
    w_progress = 1.0
    w_contact = 50.0           # only given when landing gentle
    v_target = 0.3             # desired landing speed
    a_target = 0.2             # desired landing angle (rad)
    safe_speed = 1.2           # hinge penalty if speed exceeds this
    safe_angle = 0.5           # hinge penalty if |angle| exceeds this
    w_speed_penalty = 0.5
    w_angle_penalty = 0.5

    # ----- distance progress (never hurt, guides approach) -----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = w_progress * (prev_dist - next_dist)

    # ----- contact quality (only on pad, no base gift) -----
    contact = next_obs[6] * next_obs[7]   # 1 if both legs touch
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # quality factor: 1 when perfect, 0 when speed>=v_target or angle>=a_target
    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = w_contact * contact * quality

    # ----- gentle flight penalties (only extreme cases) -----
    speed_penalty = -w_speed_penalty * max(0.0, speed - safe_speed)
    angle_penalty = -w_angle_penalty * max(0.0, angle - safe_angle)

    # ----- total (no fuel penalty) -----
    total_reward = progress + contact_reward + speed_penalty + angle_penalty

    components = {
        'progress': progress,
        'contact_reward': contact_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
    }
    return float(total_reward), components