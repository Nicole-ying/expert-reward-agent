def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress : dense signal – progress towards origin (target pad at (0,0))
      attitude_penalty   : penalty for tilt and angular velocity
      soft_landing_bonus : event-based – rewarded only on first leg contact,
                           scaled by landing quality (position, velocity, angle)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- soft landing bonus (event-based) ----------
    # Detect first leg contact
    old_any_contact = (obs[6] + obs[7]) > 0.5
    new_any_contact = (next_obs[6] + next_obs[7]) > 0.5
    landing_event = (not old_any_contact) and new_any_contact

    if landing_event:
        x = next_obs[0]            # horizontal offset from target centre
        vx = abs(next_obs[2])
        vy = abs(next_obs[3])
        angle_abs = abs(angle)

        # position factor – exponentially decay with |x|, width ~0.3
        pos_factor = 2.718281828 ** (-abs(x) / 0.3)

        # velocity penalty – low speeds desired
        vel_factor = 1.0 / (1.0 + 10.0 * (vx + vy))

        # angle penalty
        angle_factor = 1.0 / (1.0 + 5.0 * angle_abs)

        # leg factor (1 or 2)
        leg_factor = next_obs[6] + next_obs[7]  # 1 or 2

        # max bonus = 200 (aligns with environment target score)
        soft_landing_bonus = 200.0 * pos_factor * vel_factor * angle_factor * (leg_factor / 2.0)
    else:
        soft_landing_bonus = 0.0

    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components