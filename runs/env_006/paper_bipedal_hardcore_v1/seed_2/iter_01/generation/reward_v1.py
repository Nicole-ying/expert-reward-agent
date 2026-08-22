def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
    hull_angle = obs[0]             # body pitch angle
    hull_angvel = obs[1]            # body angular velocity
    horizontal_speed = obs[2]       # forward speed

    # 1. Forward progress (main learning signal)
    #    Direct dense reward for forward velocity; negative speed penalized.
    forward_progress = horizontal_speed   # coefficient = 1.0

    # 2. Balance and fall‑prevention penalty
    #    Quadratic hinge: only penalise when outside safe region.
    angle_threshold = 0.4       # rad, ~23°, safe swing allowed
    angvel_threshold = 1.0      # rad/s

    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)

    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    total_reward = forward_progress + balance_penalty

    components = {
        'forward_progress': forward_progress,
        'balance_penalty': balance_penalty
    }
    return float(total_reward), components