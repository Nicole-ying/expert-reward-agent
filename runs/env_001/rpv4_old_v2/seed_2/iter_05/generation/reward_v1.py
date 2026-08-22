def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander navigation task.
    Roles: goal_distance_progress (improvement_delta of distance to pad),
           soft_landing (proximity‑gated velocity penalty),
           orientation_penalty (quadratic on body angle),
           landing_contact_success (soft completion proxy).
    """
    # ------------------------------------------------------------------
    # Unpack observations
    # ------------------------------------------------------------------
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]
    # prev_angle = obs[4]          # not used directly
    # prev_angvel = obs[5]        # not used
    # prev_left  = obs[6], prev_right = obs[7]  # contact flags of current step are not needed

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    # next_angvel = next_obs[5]   # not used
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ------------------------------------------------------------------
    # Distance to target pad  (0,0)
    # ------------------------------------------------------------------
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # ------------------------------------------------------------------
    # 1. Goal distance progress  (improvement_delta)
    #    Positive when moving towards the pad, negative when retreating.
    # ------------------------------------------------------------------
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # ------------------------------------------------------------------
    # 2. Soft landing  (proximity‑gated velocity penalty)
    #    Large penalty for speed when close to the pad, small when far.
    # ------------------------------------------------------------------
    w_vel = 1.0
    k_dist = 20.0   # controls how sharply the gate opens near the pad
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # ------------------------------------------------------------------
    # 3. Orientation penalty  (quadratic penalty on tilt)
    #    Body angle should be near zero (upright).
    # ------------------------------------------------------------------
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # ------------------------------------------------------------------
    # 4. Soft completion proxy  (both legs on pad + very low speed)
    #    Gives a positive boost when the lander appears to have settled.
    # ------------------------------------------------------------------
    w_contact = 2.0
    v_thresh = 0.1
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    speed_low = (abs(next_vx) < v_thresh) and (abs(next_vy) < v_thresh)
    contact_bonus = w_contact if (legs_on_ground and speed_low) else 0.0

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing":          vel_penalty,
        "orientation_penalty":   angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components