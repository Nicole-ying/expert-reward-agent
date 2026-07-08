def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Bounded proximity: always provides gradient toward target
    #    (0, 1], peaks at 1 when exactly on target
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)
    
    # 2. Stability penalty: discourage violent motion, promote gentle flight
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )
    
    # 3. Continuous soft-landing proxy: product of four bounded factors [0,1]
    #    Each factor opens gradually — provides gradient before perfect conditions.
    proximity = max(0.0, 1.0 - distance / 0.5)          # activates when dist < 0.5
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)  # activates when |v| < 0.4
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)  # activates when |angle| < 0.3
    contact   = (left_contact + right_contact) / 2.0    # 0, 0.5, or 1.0
    # Floor of 0.1 ensures weak signal before any leg contact;
    # one leg = 0.55, both legs = 1.0 — gradient toward full touchdown.
    r_landing = 8.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_dist + r_stability + r_landing
    
    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components