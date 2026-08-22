def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations for current and next state
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_contact, 7: right_contact
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Compute distances from target (0,0)
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights (heuristically reasonable starting points for this environment)
    w_progress = 1.0           # scale for distance improvement
    w_speed   = 0.5            # penalty for residual kinetic energy
    w_angle   = 0.5            # penalty for tilt
    w_angvel  = 0.5            # penalty for rotation
    w_contact = 10.0           # terminal soft‑landing incentive
    alpha     = 1.0            # sharpness of vertical‑speed gate
    beta      = 1.0            # sharpness of tilt gate

    # 1. Main progress signal: approach + velocity damping
    #    Use improvement_delta for distance (reward getting closer)
    #    and penalize high speed in next state.
    progress_reward = w_progress * (dist_curr - dist_next) - w_speed * speed_next

    # 2. Orientation stabilization (safety constraint)
    orientation_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # 3. Soft‑landing contact proxy (approximate task completion)
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    # Use exponential decay to favour landing with low vertical speed and near‑vertical attitude.
    # exp(-alpha*|vy|) is computed as e ** (‑alpha * abs(vy))
    # exp(‑beta * angle²) is computed similarly.
    smooth_vy_gate = 2.718281828 ** (-alpha * abs(vy_next))
    smooth_angle_gate = 2.718281828 ** (-beta * (angle_next ** 2))
    soft_contact_reward = w_contact * both_legs_on_platform * smooth_vy_gate * smooth_angle_gate

    total_reward = progress_reward + orientation_penalty + soft_contact_reward

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_contact_reward": soft_contact_reward
    }
    return float(total_reward), components