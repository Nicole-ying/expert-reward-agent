def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    time_penalty = -0.02

    if action == 0:
        engine_penalty = 0.0
    elif action == 2:
        engine_penalty = -0.03
    else:
        engine_penalty = -0.01

    total_reward = reward_dist + reward_stability + reward_landing + time_penalty + engine_penalty

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing,
        "time_penalty": time_penalty,
        "engine_penalty": engine_penalty
    }

    return float(total_reward), components