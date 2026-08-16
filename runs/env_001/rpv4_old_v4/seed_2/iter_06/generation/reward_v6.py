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

    # --- contact guidance: transition-based (L2 change from state-value) ---
    # State-value contact reward (iter_1/iter_5) caused exploitation:
    # agent learns to occupy the platform collecting ~0.6/step indefinitely,
    # rather than settling to trigger successful termination.
    # Evidence: score peaked at cp70% (234.7) then declined to 157.2,
    # while contact_guidance rose from 27.9 to 331.9 over the same period.
    #
    # Transition-based: reward only the act of GAINING leg contact near target.
    # Once both legs are on, reward drops to zero — no incentive to linger.
    # Unlike iter_2's potential-diff (which included proximity/velocity in the
    # delta, creating 99.9% magnitude noise), this only fires on contact changes.
    contact_prev = float(obs[6] + obs[7])
    contact_next = float(next_obs[6] + next_obs[7])
    contact_gain = max(0.0, contact_next - contact_prev)   # only reward gaining legs
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)     # gate: must be near target
    contact_guidance = 100.0 * contact_gain * proximity_factor

    # --- combine all components ---
    # Progress weight=1.0 provides approach guidance.
    # Transition-based contact provides landing incentive (~200 for two legs).
    # Together they target ~200 score for a successful soft landing.
    total_reward = (1.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 1.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components