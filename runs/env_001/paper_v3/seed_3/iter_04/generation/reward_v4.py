def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet,
    staying upright and near zero velocity.
    
    Key change: approach reward is persistent bounded proximity 3/(1+dist)
    instead of potential-based delta, providing gradient even when stationary.
    """
    # Unpack next observations
    x_next, y_next = next_obs[0], next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Distance to target
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # Persistent bounded proximity: non-zero gradient even when hovering
    approach_reward = 3.0 / (1.0 + dist_next)

    # Velocity penalty gated by proximity: heavier when close
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