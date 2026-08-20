def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in Euclidean distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. descent_bonus       – reward vertical descent when horizontally aligned with pad
      D. landing_completion  – soft success bonus for two-leg contact with low speed
      E. fuel_penalty        – tiny penalty for using any engine

    v10 change: landing_completion multiplier 0.5 → 2.0.
    The wider x_gate (/0.8) from v9 solved the truncation problem (0/20 truncated),
    but caused landing quality to be undervalued — landing_completion dropped from
    70.98 (v8) to 19.17 (v9) as the agent learned to rush landings.  Increasing
    the multiplier 4× makes high-quality landings competitive with cumulative
    descent/distance signals, so the agent maintains good landing posture through
    the end of training instead of optimizing it away.
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. descent_bonus: reward vertical descent when horizontally aligned
    #    with the landing pad.  This provides continuous gradient from
    #    altitude to ground, bridging the gap between distance_progress
    #    (which is isotropic) and landing_completion (which is sparse).
    #
    #    Key design: rewards the *change* in y (delta-y), NOT the absolute
    #    y position.  Hovering at low altitude gives zero reward → no exploit.
    #    Gate ensures reward only activates when agent is above the pad.
    #
    #    x_gate /0.8 (from v9): wider than original /0.5, gives partial
    #    descent incentive to agents moderately offset horizontally,
    #    helping them descend while centering instead of requiring perfect
    #    alignment first.  This eliminated the 3/20 truncated episodes.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.8)          # 1 at x=0, 0 at |x|>=0.8
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 6.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    #
    #    v10 change: multiplier 0.5 → 2.0.  The wider descent gate caused
    #    agents to rush landings, undervaluing landing quality.  This 4×
    #    increase makes high-quality landing posture competitive with the
    #    cumulative descent/distance rewards, preventing the agent from
    #    optimizing away landing quality in late training.
    # =====================================================================
    both_feet = left_contact * right_contact            # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 2.0 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # E. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = (progress + stability_penalty + descent_bonus
                    + landing_reward + fuel_penalty)

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "descent_bonus": descent_bonus,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components