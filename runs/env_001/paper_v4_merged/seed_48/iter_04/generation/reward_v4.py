def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]

    # ---- quality based on current obs ----
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    near_goal_old = 1.0 / (1.0 + 5.0 * dist_old)
    speed_sq_old = obs[2]**2 + obs[3]**2
    low_speed_old = 1.0 / (1.0 + 10.0 * speed_sq_old)
    abs_angle_old = abs(obs[4])
    low_angle_old = 1.0 / (1.0 + 20.0 * abs_angle_old)
    contact_bonus_old = 1.0 + 2.0 * (obs[6] * obs[7])
    quality_old = near_goal_old * low_speed_old * low_angle_old * contact_bonus_old

    # ---- quality based on next_obs ----
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    near_goal_new = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq_new = next_obs[2]**2 + next_obs[3]**2
    low_speed_new = 1.0 / (1.0 + 10.0 * speed_sq_new)
    abs_angle_new = abs(next_obs[4])
    low_angle_new = 1.0 / (1.0 + 20.0 * abs_angle_new)
    contact_bonus_new = 1.0 + 2.0 * (next_obs[6] * next_obs[7])
    quality_new = near_goal_new * low_speed_new * low_angle_new * contact_bonus_new

    # improvement-only soft landing progress (prevents reward farming on a good state)
    soft_progress = max(0.0, quality_new - quality_old)

    # distance improvement
    delta_distance = dist_old - dist_new

    # engine usage penalty
    engine_penalty = 1.0 if action != 0 else 0.0

    w_dist = 10.0
    w_soft = 2.0
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components