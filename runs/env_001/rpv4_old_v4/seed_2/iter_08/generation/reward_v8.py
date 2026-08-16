def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to target (target at origin (0,0)) ---
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # --- velocity components ---
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # --- orientation penalty: body angle and angular velocity ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # --- contact guidance: state-value with comprehensive settling gates ---
    # Three gates (proximity + speed + angle) must all be satisfied simultaneously
    # to receive contact reward. This ensures reward only fires during genuine settling.
    # L1 change: coefficient 0.3 → 0.25 to slightly reduce contact dominance (was 98.9%
    # magnitude share), giving relatively more room for progress to guide the approach phase.
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0, 1.0, or 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)       # speed < 0.5 for any reward
    angle_ok = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)  # |angle| < 0.3 for any reward
    contact_guidance = 0.25 * contact_val * proximity_factor * velocity_ok * angle_ok

    # --- combine all components ---
    # L1 change: progress weight 0.5 → 0.8.
    # At 0.5, progress contributed only 0.4% magnitude share — essentially decorative.
    # At 0.8, progress provides meaningful approach-phase guidance while contact_guidance
    # (now at 0.25) remains the dominant terminal signal. The goal is to give the agent
    # more feedback during the blind approach phase, helping the 6/20 truncated episodes
    # that never reach the contact zone.
    total_reward = (0.8 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.8 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components