def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact_prev = obs[6]
    right_contact_prev = obs[7]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe landing transition bonus ------------------
    # Detect first-ever both-legs contact (binary transition 0->1)
    prev_both_contact = left_contact_prev * right_contact_prev
    curr_both_contact = next_left * next_right
    just_landed = 1.0 if (curr_both_contact > 0.5 and prev_both_contact < 0.5) else 0.0

    # Landing quality gates at the moment of first contact
    landing_dist_factor = 1.0 / (1.0 + 10.0 * next_dist)
    landing_vel_factor = 1.0 / (1.0 + 5.0 * (next_vx ** 2 + next_vy ** 2))
    landing_angle_factor = 1.0 / (1.0 + 5.0 * (abs(next_angle) + abs(next_angvel)))

    w_landing = 100.0
    safe_contact_reward = w_landing * just_landed * landing_dist_factor * landing_vel_factor * landing_angle_factor

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_reward

    reward_components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_reward": safe_contact_reward
    }

    return float(total_reward), reward_components