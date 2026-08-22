def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── distance to target (target at origin (0,0)) ──
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # ── velocity components ──
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # ── orientation penalty: body angle and angular velocity ──
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # ── contact guidance (transition-based: reward achieving contact, not lingering) ──
    contact_prev = float(obs[6] + obs[7])           # 0.0 .. 2.0
    contact_next = float(next_obs[6] + next_obs[7]) # 0.0 .. 2.0
    contact_gained = max(0.0, contact_next - contact_prev)  # fires only on 0→1 transitions

    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    # total speed must be low for safe contact (both horizontal and vertical)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)

    # (a) small per-step signal to avoid complete sparsity
    contact_dense = 0.02 * contact_next * proximity_factor * velocity_ok
    # (b) transition bonus: reward the act of making new contact
    contact_bonus = 5.0 * contact_gained * proximity_factor * velocity_ok

    contact_guidance = contact_dense + contact_bonus

    # ── combine all components ──
    total_reward = (2.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 2.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return float(total_reward), components