def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet,
    staying upright and near zero velocity.
    """
    # Unpack next_obs (post-action state)
    x = next_obs[0]          # horizontal offset from pad
    y = next_obs[1]          # vertical offset (positive = above pad)
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Distance to target (the pad is at (0,0))
    dist = (x**2 + y**2) ** 0.5

    # Positive reward that increases as distance decreases (bounded)
    approach_reward = 1.0 / (1.0 + dist)

    # Velocity penalty gated by proximity: heavy only when close
    gate_vel = 1.0 / (1.0 + dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_approach = 0.5
    w_vel      = 0.1
    comp_approach_landing = w_approach * approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle  = 0.1
    w_angvel = 0.1
    comp_stabilization = - w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    # Give a clear signal when both landing legs touch simultaneously
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components