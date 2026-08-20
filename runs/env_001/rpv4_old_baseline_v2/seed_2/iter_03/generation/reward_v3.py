def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Descent Incentive (strong, per-step downward progress) ----------
    descent = y_prev - y
    descent_reward = 10.0 * max(0.0, descent)

    # ---------- B: Target Proximity (unified position gradient toward (0,0)) ----------
    dist = (x * x + y * y) ** 0.5
    proximity_reward = 2.0 / (1.0 + dist)

    # ---------- C: Landing Quality (product of soft-landing conditions) ----------
    near_ground = 1.0 / (1.0 + 5.0 * abs(y))
    speed_quality = 1.0 / (1.0 + 3.0 * (vx * vx + vy * vy))
    attitude_quality = 1.0 / (1.0 + 2.0 * angle * angle)
    landing_quality = near_ground * speed_quality * attitude_quality

    # ---------- D: Contact Completion Bonus (large, one-time, quality-scaled) ----------
    contact = left_contact * right_contact
    contact_bonus = 60.0 * contact * speed_quality * attitude_quality

    # ---------- E: Engine Efficiency (mild) ----------
    if action == 2:
        engine_cost = -0.08
    elif action == 1 or action == 3:
        engine_cost = -0.03
    else:
        engine_cost = 0.0

    total_reward = (
        descent_reward
        + proximity_reward
        + landing_quality
        + contact_bonus
        + engine_cost
    )

    components = {
        'descent_reward': descent_reward,
        'proximity_reward': proximity_reward,
        'landing_quality': landing_quality,
        'contact_bonus': contact_bonus,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components