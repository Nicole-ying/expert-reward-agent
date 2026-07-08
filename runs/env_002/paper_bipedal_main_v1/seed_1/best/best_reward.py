def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant next observations
    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    horizontal_velocity = next_obs[2]
    vertical_velocity = next_obs[3]
    leg1_contact = next_obs[8]
    leg2_contact = next_obs[13]

    # Weights
    w_forward = 1.0
    w_angle = 1.0
    w_angvel = 0.2
    w_vert = 1.0
    w_energy = 0.05
    w_gait = 0.03

    forward_reward = w_forward * horizontal_velocity
    stability_penalty = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_velocity)
        - w_vert * (vertical_velocity ** 2)
    )
    energy_penalty = -w_energy * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )
    # Single-leg support reward: exactly one foot on ground
    single_leg_contact = float(abs(leg1_contact - leg2_contact) > 0.5)
    gait_reward = w_gait * single_leg_contact

    total_reward = forward_reward + stability_penalty + energy_penalty + gait_reward
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "gait_reward": gait_reward
    }

    return float(total_reward), components