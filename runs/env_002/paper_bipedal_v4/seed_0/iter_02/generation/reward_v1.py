def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract usable signals from the 24‑dim observation
    hv = obs[2]          # horizontal velocity (forward speed)
    angle = obs[0]       # hull angle (tilt from upright)
    leg1c = obs[8]       # leg‑1 ground contact (0/1)
    leg2c = obs[13]      # leg‑2 ground contact (0/1)

    # --- 1. Bounded forward‑velocity reward ---
    # Formula: w * x / (1 + |x|)   (bounded_signal, formula library 2.2)
    # This saturates at high speeds, preventing the agent from trying to
    # exploit unbounded velocity spikes before falling.
    w_vel = 2.0
    vel_reward = w_vel * hv / (1.0 + abs(hv))

    # --- 2. Soft health gate to penalise instability ---
    # Formula: gate = 1 / (1 + k * |angle|)   (soft_health_gate, formula library 2.6)
    # The velocity reward is scaled down when the hull tilts, turning an
    # explicit stability penalty into a continuous driving‑signal.
    k_angle = 20.0
    gate = 1.0 / (1.0 + k_angle * abs(angle))
    inst_penalty = vel_reward * (1.0 - gate)   # how much of vel_reward is gated away

    # --- 3. Gait alternation bonus ---
    # Encourages a complementary contact pattern: exactly one foot on the ground.
    # Expression = leg1 + leg2 - 2*leg1*leg2, which is 1 when contacts differ, 0 otherwise.
    w_gait = 0.2
    alternation = leg1c + leg2c - 2.0 * leg1c * leg2c
    gait_reward = w_gait * alternation

    # Total reward: velocity reward gated by stability, plus gait alternation
    total_reward = vel_reward - inst_penalty + gait_reward

    components = {
        'forward_velocity': vel_reward,
        'instability_penalty': -inst_penalty,   # shown as a negative contribution
        'gait_alternation': gait_reward
    }

    return float(total_reward), components