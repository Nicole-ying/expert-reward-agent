def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation (state after action)
    x_pos = next_obs[0]          # horizontal distance to target pad
    y_pos = next_obs[1]          # vertical distance (height) to target pad
    x_vel = next_obs[2]          # horizontal velocity
    y_vel = next_obs[3]          # vertical velocity
    body_angle = next_obs[4]     # tilt angle
    ang_vel = next_obs[5]        # angular velocity
    left_contact = next_obs[6]   # left support leg contact (0/1)
    right_contact = next_obs[7]  # right support leg contact (0/1)

    # 1. Main learning signal: distance‑based reward
    # Encourage getting close to the center of the target pad.
    distance = (x_pos**2 + y_pos**2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)        # in (0, 1], higher when closer

    # 2. Stability / smoothness penalty
    # Penalize high speeds, large tilt, and high angular velocity to promote gentle flight and soft landing.
    penalty = 0.01 * (abs(x_vel) + abs(y_vel)) + 0.1 * abs(body_angle) + 0.05 * abs(ang_vel)
    r_stability = -penalty

    # 3. Soft landing proxy (task‑completion approximation)
    # Give a one‑time bonus when the lander touches down softly on both legs near the pad center.
    landing_bonus = 0.0
    if left_contact == 1.0 and right_contact == 1.0:
        if distance < 0.2 and abs(x_vel) < 0.3 and abs(y_vel) < 0.3 and abs(body_angle) < 0.2:
            landing_bonus = 10.0

    total_reward = r_dist + r_stability + landing_bonus

    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components