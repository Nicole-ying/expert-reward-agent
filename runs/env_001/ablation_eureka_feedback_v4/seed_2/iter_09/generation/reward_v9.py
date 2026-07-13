def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Gym-style reward designed for 2D vehicle trajectory optimization.
    Focus: proximity + gated stability + continuous velocity control + contact proxy.
    """
    # --- Unpack states ---
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, nbody_angle, nang_vel, nleft_contact, nright_contact = next_obs

    # Constants
    current_sq_dist = x_pos ** 2 + y_pos ** 2
    next_sq_dist = nx_pos ** 2 + ny_pos ** 2

    PROXIMITY_SCALE = 10.0
    ANGLE_GATE_K = 15.0
    VEL_PENALTY_COEFF = 0.02
    GROUND_PROXIMITY_SCALE = 4.0
    CONTACT_PROXY_K = 4.0
    FUEL_COST_WEIGHT = 0.15

    # 1. Goal Proximity (mandatory)
    dist_improvement = current_sq_dist - next_sq_dist
    proximity_reward = PROXIMITY_SCALE * dist_improvement / (1.0 + abs(dist_improvement))

    # 2. Orientation Stability Gate
    angle_error = abs(nbody_angle)
    angle_gate = 2.718281828 ** (-ANGLE_GATE_K * angle_error ** 2)
    gated_proximity = proximity_reward * angle_gate

    # 3. Continuous Velocity Control (penalizes speed, stronger near ground)
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    ground_proximity = 1.0 / (1.0 + GROUND_PROXIMITY_SCALE * abs(ny_pos))
    velocity_penalty = -VEL_PENALTY_COEFF * (speed ** 2) * ground_proximity

    # 4. Soft Landing Contact Proxy (rewards contact near ground, no speed condition)
    leg_contact_both = min(nleft_contact, nright_contact)
    near_ground_factor = 1.0 / (1.0 + CONTACT_PROXY_K * abs(ny_pos))
    contact_proxy = (leg_contact_both * near_ground_factor) ** 0.5
    soft_landing_bonus = 2.0 * contact_proxy

    # 5. Action Efficiency
    is_engine_on = 0.0 if action == 0 else 1.0
    fuel_cost = -FUEL_COST_WEIGHT * is_engine_on

    total_reward = gated_proximity + velocity_penalty + soft_landing_bonus + fuel_cost

    components = {
        "proximity_reward": gated_proximity,
        "velocity_penalty": velocity_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_cost": fuel_cost
    }

    return float(total_reward), components