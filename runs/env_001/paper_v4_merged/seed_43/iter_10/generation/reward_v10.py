def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. progress delta: positive when approaching target
    delta_dist = dist - dist_next
    progress = 1.0 * delta_dist

    # 2. angle gate: linearly decay progress when body angle exceeds safe range
    safe_angle = 0.5  # radians
    gate_angle = max(0.3, 1.0 - abs(nangle) / safe_angle)

    # 3. contact factor: encourage both legs on ground
    if nl_contact == 1 and nr_contact == 1:
        contact_factor = 1.0
    elif nl_contact == 1 or nr_contact == 1:
        contact_factor = 0.7
    else:
        contact_factor = 0.4

    # shaped progress: main learning signal with safety and contact modulation
    shaped_progress = progress * gate_angle * contact_factor

    # 4. speed penalty near ground to promote gentle landing
    close_threshold = 0.5
    speed_penalty = 0.0
    if dist_next < close_threshold:
        speed_norm = abs(nvx) + abs(nvy)
        speed_penalty = -0.1 * speed_norm

    # 5. progress-based bonus: reward only net approach, replacing static success_bonus
    progress_bonus = 5.0 * max(0.0, delta_dist)

    # 6. action cost: small penalty for any engine use
    action_cost = -0.01 if action != 0 else 0.0

    total_reward = shaped_progress + progress_bonus + speed_penalty + action_cost

    components = {
        "progress": progress,
        "gate_angle": gate_angle,
        "contact_factor": contact_factor,
        "shaped_progress": shaped_progress,
        "speed_penalty": speed_penalty,
        "progress_bonus": progress_bonus,
        "action_cost": action_cost
    }
    return float(total_reward), components