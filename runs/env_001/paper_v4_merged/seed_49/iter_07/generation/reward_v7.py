def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1) Progress: distance reduction ----------
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist - next_dist
    progress = 1.0 * delta_dist

    # ---------- 2) Overspeed penalty ----------
    speed = (vx ** 2 + vy ** 2) ** 0.5
    safe_speed = 0.5
    overspeed = max(0.0, speed - safe_speed)
    overspeed_penalty = -0.005 * overspeed

    # ---------- 3) Landing success bonus ----------
    # Conditions: at least one leg contact, very close to origin,
    # very low speed, and upright body angle.
    contact = (left_contact > 0.5) or (right_contact > 0.5)
    dist_small = next_dist < 0.2
    speed_low = speed < 0.2
    angle_ok = abs(body_angle) < 0.3

    success_bonus = 200.0 if (contact and dist_small and speed_low and angle_ok) else 0.0

    # ---------- Total reward ----------
    total_reward = progress + overspeed_penalty + success_bonus

    components = {
        'progress': progress,
        'overspeed_penalty': overspeed_penalty,
        'success_bonus': success_bonus
    }
    return float(total_reward), components