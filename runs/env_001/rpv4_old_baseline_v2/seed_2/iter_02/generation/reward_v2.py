def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous and current state
    x_prev = obs[0]
    y_prev = obs[1]

    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Descent Incentive (main learning signal) ----------
    # Reward each step of downward progress toward the platform.
    descent = y_prev - y
    descent_reward = 3.0 * max(0.0, descent)

    # ---------- B: Altitude Proximity (auxiliary gradient near ground) ----------
    # Small, saturating reward for simply being near platform level.
    alt_reward = 0.3 / (1.0 + 3.0 * abs(y))

    # ---------- C: Horizontal Centering (gated by altitude) ----------
    # Only reward x-alignment when the craft is close to the platform.
    near_ground = 1.0 / (1.0 + 5.0 * abs(y))
    center_reward = 0.3 * near_ground / (1.0 + 2.0 * abs(x))

    # ---------- D: Soft Landing Velocity Penalty (gated) ----------
    # Penalise high speed when close to ground, with emphasis on vertical.
    speed_penalty = -1.0 * near_ground * (0.5 * vx * vx + vy * vy)

    # ---------- E: Attitude Penalty (gated by altitude) ----------
    # Penalise tilt and rotation when close to the platform.
    attitude_penalty = -0.3 * near_ground * (angle * angle + 0.5 * angvel * angvel)

    # ---------- F: Contact Success Bonus (one-time landing reward) ----------
    # Large bonus when both legs touch, scaled by landing quality.
    contact = left_contact * right_contact
    landing_quality = 1.0 / (1.0 + 10.0 * (vx * vx + vy * vy) + 3.0 * angle * angle + 3.0 * abs(x))
    contact_bonus = 25.0 * contact * landing_quality

    # ---------- G: Step Cost (anti-hovering pressure) ----------
    step_cost = -0.03

    # ---------- H: Engine Efficiency ----------
    if action == 0:
        engine_cost = 0.0
    elif action == 2:
        engine_cost = -0.05
    else:
        engine_cost = -0.02

    total_reward = (
        descent_reward
        + alt_reward
        + center_reward
        + speed_penalty
        + attitude_penalty
        + contact_bonus
        + step_cost
        + engine_cost
    )

    components = {
        'descent_reward': descent_reward,
        'altitude_proximity': alt_reward,
        'horizontal_centering': center_reward,
        'speed_penalty': speed_penalty,
        'attitude_penalty': attitude_penalty,
        'contact_bonus': contact_bonus,
        'step_cost': step_cost,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components