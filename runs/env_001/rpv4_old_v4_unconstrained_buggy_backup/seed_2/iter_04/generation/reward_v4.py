def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── distance to target (target at origin (0,0)) ──
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # ── velocity components ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed = (vx**2 + vy**2) ** 0.5

    # ── velocity penalty (same as iter_1, proven gentle and effective) ──
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor
    vel_horizontal_penalty = -0.01 * (vx ** 2)
    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # ── orientation penalty ──
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # ── contact guidance: dense reward, only near target + low speed ──
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - speed / 0.5)        # both vx and vy must be low
    contact_guidance = 0.05 * contact_val * proximity_factor * velocity_ok

    # ── combine all components ──
    progress_reward = 5.0 * progress

    total_reward = progress_reward + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": progress_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return float(total_reward), components