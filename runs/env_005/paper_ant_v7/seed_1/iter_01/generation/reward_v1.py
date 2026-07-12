def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation signals
    body_z = obs[0]                     # body height
    vx = obs[13]                        # forward velocity (world x)
    quat_x = obs[2]
    quat_y = obs[3]

    # ---- 1. Survival gate: height factor ----
    # height health: allow full reward when body_z is in [0.4, 0.8]
    # linear ramp from 0 at boundary (0.2 or 1.0) to 1 at safe zone
    low_safe  = max(0.0, min(1.0, (body_z - 0.2) / 0.2))   # distance above 0.2
    high_safe = max(0.0, min(1.0, (1.0 - body_z) / 0.2))   # distance below 1.0
    height_factor = min(low_safe, high_safe)                 # clamp by the more critical side

    # ---- 2. Survival gate: uprightness factor ----
    # compute body_up_z from quaternion: how close the body's z-axis is to world z
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, body_up_z))          # clip to [0,1], 1 = fully upright

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Light energy efficiency term ----
    w_energy = 0.0005
    action_penalty = -w_energy * sum(a**2 for a in action)

    # Total reward
    total_reward = forward_gated + action_penalty
    components = {
        'forward_gated': forward_gated,
        'action_penalty': action_penalty
    }
    return float(total_reward), components