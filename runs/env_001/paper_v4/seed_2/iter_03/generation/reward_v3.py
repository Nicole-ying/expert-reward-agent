def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant states from the next observation (post-transition)
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- Component A: Position Proximity (Main Learning Signal) ----------
    # Dense, bounded reward encouraging the craft to reach (0,0).
    dist = (x_next**2 + y_next**2) ** 0.5
    pos_reward = 1.0 / (1.0 + dist)

    # ---------- Component B: Soft Landing Velocity (Bounded Penalty) ----------
    # Penalise high horizontal/vertical velocity only when close to the ground.
    activation = 1.0 / (1.0 + 10.0 * abs(y_next))
    vel_penalty = -0.1 * activation * (vx_next**2 + vy_next**2)

    # ---------- Component C: Stable Orientation (Quadratic Penalty) ----------
    # Light penalty on body tilt to encourage horizontal attitude.
    angle_penalty = -0.5 * (angle_next ** 2)

    # ---------- Component D: Contact Completion (Non-Collapsing Joint Sum) ----------
    # Each factor is an independent bounded [0,1] measure of a desired landing condition.
    # SUM ensures each condition provides its own gradient for partial credit.
    # Coefficient reduced from 0.4 to 0.2 to lessen incentive to loiter near the pad.

    k_x = 5.0
    factor_x = 1.0 / (1.0 + k_x * x_next**2)

    k_y = 5.0
    factor_y = 1.0 / (1.0 + k_y * y_next**2)

    k_v = 2.0
    factor_v = 1.0 / (1.0 + k_v * (vx_next**2 + vy_next**2))

    k_angle = 3.0
    factor_angle = 1.0 / (1.0 + k_angle * angle_next**2)

    factor_contact = 0.5 * left_contact + 0.5 * right_contact

    contact_reward = 0.2 * (factor_x + factor_y + factor_v + factor_angle + factor_contact)

    # ---------- Total Reward ----------
    total_reward = pos_reward + vel_penalty + angle_penalty + contact_reward

    components = {
        'position_proximity': pos_reward,
        'soft_landing_velocity': vel_penalty,
        'stable_orientation': angle_penalty,
        'contact_completion': contact_reward,
    }

    return float(total_reward), components