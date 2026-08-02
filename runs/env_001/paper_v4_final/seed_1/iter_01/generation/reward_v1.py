def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x = obs[0]
    y = obs[1]
    # vx, vy not directly used for reward, but could be in future iterations
    # angle, angular_vel, contacts could be used, but here we use next_obs versions

    # Unpack next observation (state after action)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ------------------  Main progress signal (improvement_delta)  ------------------
    # Reward distance reduction to the target pad (0,0)
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = (dist - next_dist)  # positive when moving toward the target

    # -----------  Contact bonus: soft proxy for task completion  -----------
    # Sparse reward when at least one leg touches the pad and the lander is close to center
    w_contact = 10.0
    contact_threshold = 0.3
    contact_bonus = 0.0
    if (next_left_contact + next_right_contact) >= 1 and next_dist < contact_threshold:
        contact_bonus = w_contact

    # -------------------  Health constraint: body angle -------------------
    # Penalize extreme tilt that could lead to a crash (hinge form)
    w_angle = 0.5
    safe_angle = 0.5          # radians
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # -------------------  Total reward  -------------------
    total_reward = w_progress * progress + contact_bonus + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "contact_bonus": contact_bonus,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components