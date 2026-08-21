def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward progress: encourage positive horizontal velocity, penalize backward motion.
    # L1: iter_2 showed that 1.2/1.8 over-weights raw speed (agent walks faster but
    # terminates earlier and drops external score). Pull the coefficient BELOW iter_1's
    # 1.0/1.5 to suppress speed exploitation while keeping forward motion rewarded.
    if horizontal_velocity >= 0.0:
        forward_progress = 0.8 * min(horizontal_velocity, 2.0)
    else:
        forward_progress = 1.2 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Slightly strengthen torso stability so the less speed-driven gait stays upright
    # longer and does not trade stability for raw forward velocity.
    torso_stability = -0.20 * (hull_angle ** 2) - 0.07 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components