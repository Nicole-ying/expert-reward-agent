def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observations
    x = obs[0]
    y = obs[1]
    # x_vel, y_vel not directly used from obs, use next_obs for state‑transition reward
    # body_angle, angular_vel from obs are not needed, we use next_obs

    nx = next_obs[0]
    ny = next_obs[1]
    nx_vel = next_obs[2]
    ny_vel = next_obs[3]
    nbody_angle = next_obs[4]
    nangular_vel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    # Distance to target (origin) before and after the step
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # Component A: approach progress (distance reduction)
    progress = max(0.0, dist_curr - dist_next)

    # Component B: velocity penalty (continuous, on next state)
    velocity_penalty = -(nx_vel**2 + ny_vel**2)

    # Component C: angular velocity penalty (continuous, on next state)
    angular_penalty = -(nangular_vel**2)

    # Component D: soft landing quality (joint‑condition proxy)
    both_legs = nleft_contact * nright_contact          # 1.0 only when both legs contact
    speed_factor = max(0.0, 1.0 - abs(nx_vel) - abs(ny_vel))
    angle_factor = max(0.0, 1.0 - abs(nbody_angle))
    angvel_factor = max(0.0, 1.0 - abs(nangular_vel))
    landing_quality = both_legs * speed_factor * angle_factor * angvel_factor

    # Weights (tuned to make approach dominant while softly limiting velocity and oscillation)
    w_progress = 1.0
    w_vel = 0.05
    w_ang = 0.01
    w_land = 2.0

    reward_progress = w_progress * progress
    reward_vel = w_vel * velocity_penalty
    reward_ang = w_ang * angular_penalty
    reward_land = w_land * landing_quality

    total_reward = reward_progress + reward_vel + reward_ang + reward_land

    components = {
        'progress': reward_progress,
        'velocity_penalty': reward_vel,
        'angular_penalty': reward_ang,
        'landing_quality': reward_land
    }

    return float(total_reward), components