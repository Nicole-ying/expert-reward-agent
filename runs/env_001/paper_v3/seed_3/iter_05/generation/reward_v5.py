def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet.
    
    Key change: approach reward is now potential-based improvement (delta),
    not persistent state value. Hovering yields zero; only getting closer pays.
    """
    # Unpack current and next positions for potential difference
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Potential-based improvement: reward for getting closer, not for staying close
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    
    potential_curr = 5.0 / (1.0 + dist_curr)
    potential_next = 5.0 / (1.0 + dist_next)
    approach_reward = potential_next - potential_curr

    # Velocity penalty gated by proximity: heavier when close to target
    gate_vel = 1.0 / (1.0 + dist_next)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_vel = 0.1
    comp_approach_landing = approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components