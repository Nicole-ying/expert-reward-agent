def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs (post-action state)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target pad at (0,0)
    dist = (x**2 + y**2) ** 0.5

    # --- Component 1: Approach and soft landing (from best, unchanged) ---
    approach_reward = 1.0 / (1.0 + dist)
    gate_vel = 1.0 / (1.0 + dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_approach = 1.0
    w_vel = 0.1
    comp_approach_landing = w_approach * approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization (MODIFIED: distance-gated) ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    # Gate: no stabilization penalty beyond 10 units; ramps linearly to full near target
    gate_stab = max(0.0, 1.0 - dist / 10.0)
    comp_stabilization = gate_stab * (-w_angle * angle_penalty - w_angvel * angvel_penalty)

    # --- Component 3: Successful contact reward (unchanged) ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components