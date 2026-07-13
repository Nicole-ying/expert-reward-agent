def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Gym-style reward designed for lunar-lander-like 2D vehicle.
    Focus: proximity + gated stability + gentle contact proxy.
    """
    # --- Unpack states ---
    # Current (pre-step)
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact = obs

    # Next (post-step)
    nx_pos, ny_pos, nx_vel, ny_vel, nbody_angle, nang_vel, nleft_contact, nright_contact = next_obs

    # Constants
    # Square distance as convexification of linear progress (breaks local plateaus)
    current_sq_dist = x_pos ** 2 + y_pos ** 2
    next_sq_dist = nx_pos ** 2 + ny_pos ** 2

    # Gating thresholds (soft, designed for safety around landing zone)
    # 5.0 is generous, allows distant approach while heavily rewarding proximity
    PROXIMITY_SCALE = 10.0
    # 0.5 radians is ~28 degrees, beyond which gate starts to choke progress
    ANGLE_GATE_K = 15.0
    # 5.0 m/s is safe touch; hinge penalizes velocity above this linearly
    VEL_THRESH = 5.0
    # Ground proximity activation: velocity penalty scales with 1/y_pos when close
    VEL_PENALTY_DIST_SCALE = 8.0
    # Contact proxy smoothing
    CONTACT_PROXY_K = 4.0
    # Fuel/action cost (mild)
    FUEL_COST_WEIGHT = 0.15

    # 1. Goal Proximity (mandatory) — convexified distance progress
    # Uses improvement_delta on squared distance: creates strong gradient near origin
    dist_improvement = current_sq_dist - next_sq_dist  # positive means got closer
    # Compress improvement to avoid spam rewards when already at goal
    proximity_reward = PROXIMITY_SCALE * dist_improvement / (1.0 + abs(dist_improvement))

    # 2. Orientation Stability Gate (soft health gate for primary progress)
    # The idea: posture error attenuates proximity_reward. If agent is tilted, it can't
    # collect full progress — this forces posture correction *before* arrival, not after.
    angle_error = abs(nbody_angle)  # 0 is ideal horizontal
    # Exponential gate: near 1.0 when angle small, falls rapidly beyond 0.05 rad
    angle_gate = 2.718281828 ** (-ANGLE_GATE_K * angle_error ** 2)
    gated_proximity = proximity_reward * angle_gate

    # 3. Soft Landing Velocity Penalty (conditional on proximity to ground)
    # Only meaningful when near surface (small ny_pos). Avoid penalizing fast descent
    # from high altitude.
    # We use a soft hinge on speed magnitude, weighted by ground proximity.
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    # Hinge: only penalizes speed above VEL_THRESH, linearly
    excess_speed = max(0.0, speed - VEL_THRESH)
    # Ground proximity weight: 1/(1 + scale * ny_pos) → peaks when close to ground
    ground_proximity = 1.0 / (1.0 + VEL_PENALTY_DIST_SCALE * abs(ny_pos))
    velocity_penalty = -excess_speed * ground_proximity

    # 4. Soft Landing Proxy (joint condition with continuous factors)
    # Encourages simultaneous: both legs + low speed + near ground.
    # Each factor is smooth and continuous, avoids binary collapse.
    leg_contact_both = min(nleft_contact, nright_contact)  # 0 or 1, mostly
    slow_factor = 1.0 / (1.0 + CONTACT_PROXY_K * speed)
    near_ground_factor = 1.0 / (1.0 + 4.0 * abs(ny_pos))
    # Geometric mean of three factors to avoid product collapse
    contact_proxy = (leg_contact_both * slow_factor * near_ground_factor) ** (1.0 / 3.0)
    soft_landing_bonus = 2.0 * contact_proxy

    # 5. Action Efficiency (sparse penalty on engine use)
    # Action 2 = main engine, 1 and 3 = orientation engines.
    # We penalize any non-zero action, not just magnitude.
    is_engine_on = 0.0 if action == 0 else 1.0
    fuel_cost = -FUEL_COST_WEIGHT * is_engine_on

    # Assemble total reward
    total_reward = gated_proximity + velocity_penalty + soft_landing_bonus + fuel_cost

    # Components dict for introspection
    components = {
        "proximity_reward": gated_proximity,
        "velocity_penalty": velocity_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_cost": fuel_cost
    }

    return float(total_reward), components