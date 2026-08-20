def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    a_v = 10.0       # sensitivity for vertical speed
    b_angle = 10.0   # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling and crash
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    total_reward = progress + orientation_penalty + landing_quality

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components