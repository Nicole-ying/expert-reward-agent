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

    # --- contact guidance: state-based occupancy reward ---
    # Reverted from potential-based difference (iter_2/iter_3) back to the
    # state-value formula proven in iter_1 (score=171). The potential-diff
    # approach is fundamentally broken for discrete contact signals:
    # contact_sum * prox * v_ok = 0 whenever legs are off the ground,
    # making pot_next - pot_prev = 0 in >99% of steps (active_rate=0.9%).
    #
    # State-based reward gives dense positive signal whenever the agent is
    # near the platform, slow, and has legs touching — guiding it to
    # discover and sustain contact.
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok

    # --- combine all components ---
    # Progress weight kept at 5.0 (from iter_3) to provide stronger approach
    # signal than iter_1's 0.5, while contact_guidance remains the dominant
    # terminal guidance channel.
    total_reward = (5.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 5.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components