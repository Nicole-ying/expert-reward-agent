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
    # Uses soft inverse distance so that reward saturates near the target.
    dist = (x_next**2 + y_next**2) ** 0.5
    pos_reward = 1.0 / (1.0 + dist)

    # ---------- Component B: Soft Landing Velocity (Bounded Penalty) ----------
    # Penalise high horizontal/vertical velocity only when close to the ground.
    # Activation gate: high when |y| is small (near platform level), negligible in high sky.
    activation = 1.0 / (1.0 + 10.0 * abs(y_next))
    vel_penalty = -0.1 * activation * (vx_next**2 + vy_next**2)

    # ---------- Component C: Stable Orientation (Quadratic Penalty) ----------
    # Light penalty on body tilt to encourage horizontal attitude.
    angle_penalty = -0.5 * (angle_next ** 2)

    # ---------- Component D: Contact Completion (Joint Condition Proxy) ----------
    # Soft proxy for a “successful two‑leg landing” when multiple conditions are met.
    # Each factor is a continuous [0,1] measure of a desired condition.
    k_x = 5.0
    factor_x = 1.0 / (1.0 + k_x * x_next**2)
    k_y = 5.0
    factor_y = 1.0 / (1.0 + k_y * y_next**2)
    k_vx = 2.0
    factor_vx = 1.0 / (1.0 + k_vx * vx_next**2)
    k_vy = 2.0
    factor_vy = 1.0 / (1.0 + k_vy * vy_next**2)
    k_angle = 3.0
    factor_angle = 1.0 / (1.0 + k_angle * angle_next**2)
    # Contact signals are binary; the product is non‑zero only when both legs touch.
    factor_contact = left_contact * right_contact

    contact_proxy = factor_x * factor_y * factor_vx * factor_vy * factor_angle * factor_contact
    contact_reward = 1.0 * contact_proxy   # up to 1.0 when all conditions are nearly perfect

    # ---------- Total Reward ----------
    total_reward = pos_reward + vel_penalty + angle_penalty + contact_reward

    components = {
        'position_proximity': pos_reward,
        'soft_landing_velocity': vel_penalty,
        'stable_orientation': angle_penalty,
        'contact_completion': contact_reward,
    }

    return float(total_reward), components