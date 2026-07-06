def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs (outcome of action)
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    torso_ang_vel = next_obs[7]        # rad/s

    # Forward progress: reward positive forward velocity unconditionally
    forward_reward = max(0.0, forward_vel)

    # Uprightness bonus: independent additive reward for staying near-upright
    temp = 0.5
    upright_bonus = 2.718281828 ** (-abs(torso_angle) / temp)

    # Stability penalty: light quadratic penalty on tilt and angular velocity
    stability_penalty = -0.1 * (torso_angle ** 2) - 0.01 * (torso_ang_vel ** 2)

    total_reward = forward_reward + upright_bonus + stability_penalty

    components = {
        'forward_reward': forward_reward,
        'upright_bonus': upright_bonus,
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components