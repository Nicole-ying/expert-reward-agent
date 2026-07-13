def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # extract current and next state
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # distances from target
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # progress (positive if getting closer, negative if drifting away)
    progress = prev_dist - next_dist
    c_progress = 5.0 * progress   # stronger bidirectional shaping

    # contact reward
    contact_count = n_left_contact + n_right_contact
    c_contact = 2.0 * contact_count

    # posture penalties
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.01 * (n_angvel**2)

    # per-step alive penalty to encourage faster completion
    c_alive = -0.005

    # fuel penalty for using main engine (action 2)
    fuel_penalty = -0.02 if (action == 2) else 0.0

    # soft-landing bonus (unchanged, only when both legs contact)
    two_legs = (n_left_contact > 0.5 and n_right_contact > 0.5)
    if two_legs:
        dist = next_dist
        speed_mag = (nvx**2 + nvy**2) ** 0.5
        angle_mag = abs(n_angle)
        angvel_mag = abs(n_angvel)

        score_dist = max(0.0, 1.0 - dist / 0.3)
        score_speed = max(0.0, 1.0 - speed_mag / 0.3)
        score_angle = max(0.0, 1.0 - angle_mag / 0.2)
        score_angvel = max(0.0, 1.0 - angvel_mag / 0.3)

        c_landing = 200.0 * score_dist * score_speed * score_angle * score_angvel
    else:
        c_landing = 0.0

    total_reward = (c_progress + c_contact + c_angle + c_angvel +
                    c_landing + c_alive + fuel_penalty)

    components = {
        'distance_progress': c_progress,
        'contact_reward': c_contact,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
        'landing_bonus': c_landing,
        'alive_penalty': c_alive,
        'main_engine_penalty': fuel_penalty,
    }
    return (float(total_reward), components)