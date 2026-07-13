def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
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

    # ------------------ 4. Safe contact bonus (transition-based) ------------------
    w_contact = 10.0
    prox_factor = 1.0 / (1.0 + 10.0 * next_dist)
    contact_before = obs[6] * obs[7]         # 1.0 if both legs already touching
    contact_now = next_left * next_right      # 1.0 if both legs touching after step
    contact_transition = (1.0 - contact_before) * contact_now  # 1.0 only on newly achieved dual contact
    safe_contact_bonus = w_contact * prox_factor * contact_transition

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }

    return float(total_reward), components