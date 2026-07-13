def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state (post-action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: State-based proximity reward ---
    # Continuous per-step reward that grows as distance to pad shrinks.
    # Bounded above at 1.0 (when dist=0), provides gradient at all distances.
    curr_dist = (x**2 + y**2) ** 0.5
    w_approach = 1.0
    comp_approach = w_approach / (1.0 + curr_dist)

    # --- Component 2: Soft landing velocity penalty (distance-gated) ---
    # Gate: full strength only when close to pad.
    gate_vel = 1.0 / (1.0 + curr_dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_vel = 0.2  # increased from 0.1 to strengthen soft-landing guidance
    comp_soft_landing = -w_vel * vel_penalty

    # --- Component 3: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 4: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach + comp_soft_landing + comp_stabilization + comp_contact
    components = {
        "approach_proximity": comp_approach,
        "soft_landing_velocity": comp_soft_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components