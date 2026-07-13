def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- weights & thresholds -----
    w_progress = 10.0
    w_contact_base = 5.0
    w_contact_quality = 40.0
    v_target = 0.5          # speed where quality reaches 0
    a_target = 0.3          # angle (rad) where quality reaches 0
    w_speed_penalty = 0.1
    w_angle_penalty = 0.1
    safe_speed = 0.8        # no penalty below this speed
    safe_angle = 0.3        # no penalty below this angle (rad)
    w_fuel = 0.02           # tiny fuel penalty per firing step

    # ----- 1. distance progress (no gate) -----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist
    progress_reward = w_progress * progress

    # ----- 2. contact reward (base + quality) -----
    contact = next_obs[6] * next_obs[7]  # 1.0 only when both legs on pad
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # linear quality factors: 1 -> 0 as speed/angle approach v_target/a_target
    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = w_contact_base * contact + w_contact_quality * contact * quality

    # ----- 3. hinge penalties (protect against too fast / too tilted) -----
    speed_penalty = -w_speed_penalty * max(0.0, speed - safe_speed)
    angle_penalty = -w_angle_penalty * max(0.0, angle - safe_angle)

    # ----- 4. fuel penalty (discrete action) -----
    fuel_penalty = -w_fuel * (1.0 if action != 0 else 0.0)

    # ----- total -----
    total_reward = progress_reward + contact_reward + speed_penalty + angle_penalty + fuel_penalty

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components