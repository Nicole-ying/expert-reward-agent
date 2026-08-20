def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Approach (potential-based, no living reward) ----------
    # Reward MOVING toward (0,0), not BEING near (0,0).
    # Zero reward for hovering — removes the "hover and collect" exploit.
    prev_dist = (x_prev * x_prev + y_prev * y_prev) ** 0.5
    curr_dist = (x * x + y * y) ** 0.5
    approach_reward = 5.0 * (prev_dist - curr_dist)

    # ---------- B: Descent (strong, per-step downward progress) ----------
    descent = y_prev - y
    descent_reward = 20.0 * max(0.0, descent)

    # ---------- C: Soft Landing Velocity Penalty (gated by height) ----------
    near_ground = 1.0 / (1.0 + 3.0 * abs(y))
    vel_penalty = -5.0 * near_ground * (vx * vx + vy * vy)

    # ---------- D: Attitude Penalty (gated by height) ----------
    attitude_penalty = -2.0 * near_ground * (angle * angle + 0.3 * angvel * angvel)

    # ---------- E: Contact Reward (discoverable gradient via sum, not product) ----------
    contact_sum = left_contact + right_contact
    contact_quality = 1.0 / (1.0 + 5.0 * (vx * vx + vy * vy) + 5.0 * angle * angle + 3.0 * abs(x))
    contact_reward = 80.0 * contact_sum * contact_quality

    # ---------- F: Engine Cost (mild) ----------
    if action == 2:
        engine_cost = -0.05
    elif action == 1 or action == 3:
        engine_cost = -0.02
    else:
        engine_cost = 0.0

    total_reward = (
        approach_reward
        + descent_reward
        + vel_penalty
        + attitude_penalty
        + contact_reward
        + engine_cost
    )

    components = {
        'approach_reward': approach_reward,
        'descent_reward': descent_reward,
        'vel_penalty': vel_penalty,
        'attitude_penalty': attitude_penalty,
        'contact_reward': contact_reward,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components