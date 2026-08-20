def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs and next_obs are both length 8 vectors
    # indices: 0:x_position, 1:y_position, 2:x_velocity, 3:y_velocity,
    #          4:body_angle, 5:angular_velocity, 6:left_support_contact, 7:right_support_contact
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # Current and next distances to target pad (0,0)
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward (main learning signal) ------------------
    w_progress = 2.0
    progress = dist - next_dist          # positive when moving closer
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated constraint) ------------------
    gate = 1.0 / (1.0 + dist)            # weak when far, strong when close
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty (efficiency) ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact bonus (soft completion proxy) ------------------
    w_contact = 0.5
    # continuous proximity factor: 1 when distance -> 0, ~0 when distance large
    prox_factor = 1.0 / (1.0 + 10.0 * next_dist)
    # only active when both legs touch
    safe_contact_bonus = w_contact * prox_factor * next_left * next_right

    # ---------------------------------------------------------------------------------
    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }

    return float(total_reward), components