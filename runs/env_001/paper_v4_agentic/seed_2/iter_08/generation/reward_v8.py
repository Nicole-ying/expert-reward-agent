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
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    norm_distance = distance_to_target / 2.5
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential

    # --- Component B: soft velocity health gate (unchanged) ---
    speed = (vx**2 + vy**2) ** 0.5
    safe_speed = 0.3 + 1.5 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    gate = 0.3 + 0.7 * (2.718281828 ** (-max(0.0, overspeed_ratio - 1.0)))

    scaled_progress = potential_delta * 10.0
    gated_progress = scaled_progress * gate

    # --- Component C: landing steady-state reward (REVISED thresholds) ---
    # Distance factor: linear decay, zero beyond 0.25 (slightly relaxed from 0.15)
    dist_factor = max(0.0, 1.0 - distance_to_target / 0.25)

    # Contact factor: requires BOTH legs on pad
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0

    # Speed factor: relaxed from 0.1 to 0.3 threshold
    # Soft landing below 0.3 is acceptable; linear decay to 0 at 0.5
    if speed < 0.3:
        speed_factor = 1.0
    else:
        speed_factor = max(0.0, 1.0 - (speed - 0.3) / 0.2)

    # Angle factor: relaxed from 0.1 to 0.25 threshold
    # Near-upright below 0.25 rad (~14 deg) is good enough
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.25)

    # Angular velocity factor: relaxed from 0.5 to 0.8 threshold
    angular_factor = max(0.0, 1.0 - abs(angular_vel) / 0.8)

    # Product — any factor can zero out if far from target
    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.15 * landing_factor  # reduced from 0.3 due to higher active_rate

    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components