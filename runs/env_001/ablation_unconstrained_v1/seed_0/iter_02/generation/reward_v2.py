def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x_pos = obs[0]
    y_pos = obs[1]

    # Next state
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Potential-based progress: reward getting closer, penalize moving away
    #    No reward for staying still. Total over episode is bounded by initial distance.
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    r_progress = 10.0 * (dist_current - dist_next)

    # 2. Continuous landing proxy: replaces hard binary bonus with smooth gradient
    #    Each factor is bounded in [0, 1]; product gives dense learning signal.
    proximity = max(0.0, 1.0 - dist_next / 0.3)
    stillness = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 0.5)
    upright = max(0.0, 1.0 - abs(body_angle_next) / 0.3)
    # Contact factor: 0.15 floor ensures weak signal before any leg contact;
    # one leg = 0.575, both legs = 1.0 — strong gradient toward touchdown.
    contact_factor = 0.15 + 0.85 * (left_contact + right_contact) / 2.0
    r_landing = 30.0 * proximity * stillness * upright * contact_factor

    # 3. Time penalty: discourage hovering, incentivize fast completion
    r_time = -0.02

    total_reward = r_progress + r_landing + r_time

    components = {
        "progress_reward": r_progress,
        "landing_proxy": r_landing,
        "time_penalty": r_time
    }

    return float(total_reward), components