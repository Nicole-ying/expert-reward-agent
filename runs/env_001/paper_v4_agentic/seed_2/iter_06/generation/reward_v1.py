def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Extract next_obs signals ---
    x = next_obs[0]          # horizontal position relative to pad center
    y = next_obs[1]          # vertical height above pad
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body angle (0 upright)
    angular_vel = next_obs[5]  # angular velocity
    left_contact = next_obs[6]  # left leg contact
    right_contact = next_obs[7] # right leg contact

    # --- Helper: distance from pad center (target) ---
    horizontal_dist = abs(x)
    # Euclidean distance to target (x=0, y=0 meaning on the pad)
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    # Potential: combines position error and orientation error.
    # At the goal (upright, centered, low altitude), potential → 0 (highest potential).
    # We define potential as negative distance minus an angle penalty.
    # Normalize distance: max plausible distance ~ sqrt(1.5^2 + 2.0^2) ≈ 2.5, so /2.5 gives ~[0,1]
    norm_distance = distance_to_target / 2.5
    # Angle penalty: absolute angle, normalized by pi (~1.57 rad max before crash)
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    # Previous potential (from obs)
    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential
    A_progress = potential_delta  # scale ~ [-0.2, 0.2] per step

    # --- Component B: soft velocity health gate on approach to pad ---
    # Agent must slow down as it gets close. If speed is too high relative to distance,
    # we attenuate the progress reward.
    # Speed norm: sqrt(vx^2 + vy^2). Typical safe landing speed < 0.5.
    speed = (vx**2 + vy**2) ** 0.5
    # Gate threshold: at distance d, safe speed ≈ 0.3 + 0.7*d. If speed >> safe, gate → 0.3.
    safe_speed = 0.3 + 0.7 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    # Gate factor: 1 when ratio ≤ 1, decays to 0.3 when ratio ≥ 2.5 using smooth exponential
    speed_gate = 0.3 + 0.7 * (2.718281828 ** (-2.0 * max(0, overspeed_ratio - 1.0)**2))

    # Apply gate to progress reward
    gated_progress = A_progress * speed_gate

    # --- Component C: landing steady-state reward ---
    # When the lander is near the pad center, upright, with legs contact, and nearly still,
    # it should receive a continuous positive signal to reinforce staying there.
    # Conditions:
    #   - distance < 0.15 (close to center)
    #   - both legs in contact
    #   - speed < 0.1
    #   - |angle| < 0.1 rad
    # Use a product of soft bounded factors to avoid 0/1 cliff.
    dist_factor = max(0, 1.0 - distance_to_target / 0.15)
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    speed_factor = max(0, 1.0 - speed / 0.1)
    angle_factor = max(0, 1.0 - abs(angle) / 0.1)
    angular_factor = max(0, 1.0 - abs(angular_vel) / 0.5)

    # Product of factors (only active when all are > 0)
    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.3 * landing_factor  # modest bonus, does not dominate

    # --- Total reward ---
    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components