def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # next_obs
    nx = next_obs[0]
    ny = next_obs[1]

    # Hyperparameters
    w_goal = 5.0                    # potential-based progress weight
    alpha_proximity = 5.0           # controls the activation radius for proximity-based terms
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 8.0

    # Squared distances for potential computation
    dist_sq_obs = x**2 + y**2
    dist_sq_next = nx**2 + ny**2

    # Soft proximity weight based on current position
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq_obs)

    # 1. Main progress: potential-based improvement toward target center
    #    Positive when moving closer, negative when moving away, bounded total
    goal_proximity = w_goal * (dist_sq_obs - dist_sq_next)

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: encourage both legs touching, gated by proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components