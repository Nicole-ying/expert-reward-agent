def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: goal-directed velocity alignment (dense, primary)
    # ------------------------------------------------------------
    pos_x, pos_y = obs[0], obs[1]
    vel_x, vel_y = obs[2], obs[3]
    # Dot product of position and velocity: positive = moving outward,
    # negative = moving toward origin. We reward moving toward origin.
    dot = pos_x * vel_x + pos_y * vel_y
    # Bounded form to keep scale in [-1, 1]
    direction_reward = (-dot) / (1.0 + abs(dot))

    # ------------------------------------------------------------
    # Component B: soft landing proxy (continuous, bounded)
    # ------------------------------------------------------------
    body_angle = obs[4]
    # Contact factor: average contact, encourages both feet
    contact_factor = (obs[6] + obs[7]) / 2.0
    # Angle factor: 1 when upright (angle 0), 0 when |angle| > 0.5 rad
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.5)
    # Speed factor: 1 when near zero speed, 0 when speed^2 > 0.5
    speed2 = vel_x**2 + vel_y**2
    speed_factor = max(0.0, 1.0 - speed2 / 0.5)
    # Joint product proxy for settled landing
    soft_landing_proxy = contact_factor * angle_factor * speed_factor

    # ------------------------------------------------------------
    # Component C: contact event bonus (one-shot settle detection)
    # ------------------------------------------------------------
    event_bonus = 0.0
    # Both feet on ground in next_obs, but not in current obs
    if next_obs[6] == 1.0 and next_obs[7] == 1.0 and (obs[6] == 0.0 or obs[7] == 0.0):
        n_vel_x, n_vel_y = next_obs[2], next_obs[3]
        n_angle = next_obs[4]
        # Require low speed and near-upright to count as a genuine settle event
        if (n_vel_x**2 + n_vel_y**2) < 0.5 and abs(n_angle) < 0.5:
            event_bonus = 10.0

    # ------------------------------------------------------------
    # Component D: attitude stability penalty
    # ------------------------------------------------------------
    ang_vel = obs[5]
    attitude_penalty = -0.1 * (abs(body_angle) + abs(ang_vel))

    # ------------------------------------------------------------
    # Component E: light fuel penalty (mandatory role)
    # ------------------------------------------------------------
    fuel_penalty = -0.01 if action != 0 else 0.0

    # ------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------
    total_reward = (1.0 * direction_reward +
                    0.5 * soft_landing_proxy +
                    event_bonus +
                    attitude_penalty +
                    fuel_penalty)

    components = {
        "direction_reward": direction_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "event_bonus": event_bonus,
        "attitude_penalty": attitude_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components