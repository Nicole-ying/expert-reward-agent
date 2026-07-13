def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = obs[0]          # horizontal position relative to pad
    y = obs[1]          # vertical position relative to pad
    vx = obs[2]         # horizontal velocity
    vy = obs[3]         # vertical velocity
    angle = obs[4]      # body angle (rad)
    left_contact = obs[6]   # left leg contact flag
    right_contact = obs[7]  # right leg contact flag

    # --- Component A: goal-approach (scaled & gated) ---
    distance = (x**2 + y**2)**0.5
    goal_reward = 1.0 / (1.0 + 5.0 * distance)

    # Soft gate on speed and angle (slightly relaxed)
    speed_sq = vx**2 + vy**2
    speed_gate = 1.0 / (1.0 + 1.0 * speed_sq)   # relaxed from 2.0
    angle_gate = 1.0 / (1.0 + 0.5 * abs(angle))  # relaxed from 1.0
    gate = speed_gate * angle_gate
    gated_goal = goal_reward * gate

    # --- Component B: height incentive (encourage descent) ---
    # Give reward when y is small (close to pad). Bounded [0, 1].
    height_reward = 1.0 / (1.0 + 3.0 * abs(y))

    # --- Component C: contact bonus (stronger landing incentive) ---
    # Boost for both legs touching.
    contact_bonus = 2.0 * (left_contact * right_contact)  # 2.0 if both touch, else 0

    # --- Component D: engine usage penalty (fuel saving) ---
    # Penalize any action that fires an engine (action != 0)
    engine_penalty = -0.05 if action != 0 else 0.0

    # Total reward
    total_reward = gated_goal + height_reward + contact_bonus + engine_penalty

    components = {
        'gated_goal': gated_goal,
        'height_reward': height_reward,
        'contact_bonus': contact_bonus,
        'engine_penalty': engine_penalty
    }

    return float(total_reward), components