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

    next_x = next_obs[0]
    next_y = next_obs[1]

    # Hyperparameters
    w_goal = 1.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 3.0

    # Distance to target center (squared) for current and next state
    dist_sq = x**2 + y**2
    next_dist_sq = next_x**2 + next_y**2

    # Narrow proximity gate for contact reward: only reward leg contact very close to target
    proximity_narrow = 1.0 / (1.0 + 5.0 * dist_sq)

    # Wide proximity gate for velocity penalty: provide deceleration feedback earlier on approach
    proximity_wide = 1.0 / (1.0 + 1.5 * dist_sq)

    # 1. Main progress: potential-based improvement reward (positive when approaching target)
    goal_proximity = w_goal * (dist_sq - next_dist_sq)

    # 2. Soft landing velocity penalty: active over a wider approach zone
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity_wide

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: reward both legs touching the pad, gated by narrow proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity_narrow

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components