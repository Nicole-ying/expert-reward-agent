def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    Components:
        A: progress (distance reduction towards target)  -- main learning signal (STRENGTHENED)
        B: velocity damping (next_obs squared velocity)  -- stability constraint
        C: orientation stabilization (next_obs angle/angular velocity) -- stability constraint
        D: fuel penalty (per action cost)                -- efficiency / mild action cost
        E: success_landing_bonus (transition-based)      -- terminal success signal, one-time
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 20.0  # iter_10: 2.0 → 20.0 (10x) to counter success_landing_bonus drowning
    component_A = w_progress * progress

    # Component B: velocity damping – penalise large speeds in next step
    vx_n, vy_n = next_obs[2], next_obs[3]
    vel_penalty = vx_n**2 + vy_n**2
    w_vel = 0.05
    component_B = -w_vel * vel_penalty

    # Component C: orientation stabilization – penalise tilt and rotation
    angle_n, angvel_n = next_obs[4], next_obs[5]
    ang_penalty = angle_n**2 + angvel_n**2
    w_ang = 0.05
    component_C = -w_ang * ang_penalty

    # Component D: fuel penalty – light cost for any engine use
    fuel_cost = 0.0 if action == 0 else 1.0
    w_fuel = 0.01
    component_D = -w_fuel * fuel_cost

    # Component E: success landing bonus – one-time bonus on foot contact transition
    # Trigger: both support legs transition from OFF to ON in a single step
    # AND soft-landing conditions are met (moderate thresholds to avoid dead component)
    left_contact = obs[6]
    right_contact = obs[7]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    prev_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    contact_made = curr_both_contact and (not prev_both_contact)

    # Soft landing conditions: near target, slow, upright
    landing_dist = (nx**2 + ny**2) ** 0.5
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    is_soft_landing = (landing_dist < 0.4) and (landing_speed < 1.0) and (landing_tilt < 0.5)

    w_success = 100.0
    component_E = w_success if (contact_made and is_soft_landing) else 0.0

    total_reward = component_A + component_B + component_C + component_D + component_E

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D,
        "success_landing_bonus": component_E
    }

    return float(total_reward), components