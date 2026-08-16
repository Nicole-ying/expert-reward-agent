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

    # --- contact guidance: potential-based difference reward ---
    # Reward IMPROVEMENT in contact state, not occupancy of it.
    # This eliminates the plateau where the agent can hover near the platform
    # collecting rewards indefinitely without fully settling.
    def contact_potential(dist, vy_val, contact_sum):
        """Potential function: higher when closer, slower, and more legs on ground."""
        prox = max(0.0, 1.0 - dist / 5.0)
        v_ok = max(0.0, 1.0 - abs(vy_val) / 0.5)
        return contact_sum * prox * v_ok

    contact_prev = float(obs[6] + obs[7])
    contact_next = float(next_obs[6] + next_obs[7])

    pot_prev = contact_potential(dist_prev, obs[3], contact_prev)
    pot_next = contact_potential(dist_next, next_obs[3], contact_next)

    # Multiplier reduced from 100.0 to 3.0 to stop dominating training with
    # large-magnitude oscillations (was 99.9% magnitude_share, 2.8% signed_share).
    contact_guidance = 3.0 * (pot_next - pot_prev)

    # --- combine all components ---
    # Progress weight increased from 2.0 to 5.0 to strengthen the approach signal
    # now that contact_guidance no longer monopolizes the gradient.
    total_reward = (5.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 5.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components