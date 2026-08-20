def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack relevant observations
    x = obs[0]          # horizontal distance to target pad
    y = obs[1]          # vertical distance (height) above pad
    x_vel = obs[2]      # horizontal velocity
    y_vel = obs[3]      # vertical velocity (positive upward)
    angle = obs[4]      # body pitch angle
    left_contact = obs[6]
    right_contact = obs[7]

    # 1. Distance penalty – unbounded, drives x and y to zero
    dist_penalty = -0.2 * (abs(x) + abs(y))

    # 2. Velocity penalty – encourages low speeds for soft landing
    vel_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 3. Angle penalty – keeps the vehicle upright
    ang_penalty = -0.5 * abs(angle)

    # 4. Continuous contact-stability reward – activated by leg contact,
    #    scaled by how stable the arrival is.
    contact_factor = (left_contact + right_contact) / 2.0
    # stability is high when speed and angle are small
    stability = 2.0 / (1.0 + abs(x_vel) + abs(y_vel) + 10.0 * abs(angle))
    landing_reward = 5.0 * contact_factor * stability

    # 5. Fuel efficiency penalty – small disincentive for any engine use
    fuel_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    # Total reward
    total_reward = dist_penalty + vel_penalty + ang_penalty + landing_reward + fuel_penalty

    components = {
        "dist_penalty": dist_penalty,
        "vel_penalty": vel_penalty,
        "ang_penalty": ang_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components