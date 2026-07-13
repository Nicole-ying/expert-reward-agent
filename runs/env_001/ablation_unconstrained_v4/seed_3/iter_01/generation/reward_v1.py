def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = obs[0]          # horizontal position relative to pad
    y = obs[1]          # vertical position relative to pad
    vx = obs[2]         # horizontal velocity
    vy = obs[3]         # vertical velocity
    angle = obs[4]      # body angle (rad)
    left_contact = obs[6]   # left leg contact flag
    right_contact = obs[7]  # right leg contact flag

    # === Component A: goal-approach reward ===
    # Dense, positive, bounded reward for being close to the pad center.
    distance = (x**2 + y**2)**0.5
    goal_reward = 1.0 / (1.0 + 5.0 * distance)

    # === Component B: soft-landing gate ===
    # Multiplicative gate that dampens goal reward when speed or angle are high.
    speed_sq = vx**2 + vy**2
    speed_gate = 1.0 / (1.0 + 2.0 * speed_sq)
    angle_gate = 1.0 / (1.0 + 1.0 * abs(angle))
    gate = speed_gate * angle_gate

    gated_goal_reward = goal_reward * gate

    # === Component C: leg-contact bonus ===
    # Direct encouragement for both legs touching the pad.
    contact_reward = 0.2 * (left_contact + right_contact)

    total_reward = gated_goal_reward + contact_reward

    components = {
        'gated_goal_reward': gated_goal_reward,
        'contact_reward': contact_reward
    }

    return float(total_reward), components