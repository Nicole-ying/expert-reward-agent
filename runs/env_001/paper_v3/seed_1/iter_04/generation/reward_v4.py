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
    nvx = next_obs[2]
    nvy = next_obs[3]
    n_angle = next_obs[4]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # Hyperparameters
    w_goal = 1.0
    alpha_proximity = 5.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 50.0          # raised for one-time landing bonus
    beta_speed = 10.0          # speed quality decay: 1/(1+beta_speed*speed_sq)
    beta_angle = 10.0          # angle quality decay: 1/(1+beta_angle*angle_sq)

    # Distance to target center (squared) for current state
    dist_sq = x**2 + y**2

    # Soft proximity weight for velocity penalty (uses current position)
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: one-time landing bonus on transition to dual-leg contact,
    #    gated by proximity, speed quality, and angle quality at the moment of landing.
    obs_dual = left_contact * right_contact
    next_dual = n_left_contact * n_right_contact

    # Rising edge: newly achieved dual-leg contact (0->1 transition)
    landing_transition = max(0.0, next_dual - obs_dual)

    # Quality gates based on next_obs (state at landing moment)
    dist_sq_next = nx**2 + ny**2
    proximity_next = 1.0 / (1.0 + alpha_proximity * dist_sq_next)
    speed_quality = 1.0 / (1.0 + beta_speed * (nvx**2 + nvy**2))
    angle_quality = 1.0 / (1.0 + beta_angle * (n_angle**2))

    contact_reward = w_contact * landing_transition * proximity_next * speed_quality * angle_quality

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components