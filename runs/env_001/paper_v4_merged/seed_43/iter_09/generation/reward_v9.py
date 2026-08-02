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

    # 5. continuous success proxy using geometric mean of proximity, stability and contact
    proximity = 1.0 / (1.0 + 10.0 * dist_next)
    speed_norm_eucl = (nvx**2 + nvy**2) ** 0.5
    stability = 1.0 / (1.0 + 3.0 * speed_norm_eucl + 3.0 * abs(nangle))
    contact_quality = (nl_contact + nr_contact) / 2.0  # in [0,1]
    # geometric mean to avoid product collapse; add tiny epsilon for zero case
    product = proximity * stability * contact_quality
    eps = 1e-6
    success_factor = (max(product, eps)) ** (1.0 / 3.0)
    success_bonus = 5.0 * success_factor

    # 6. action cost: small penalty for any engine use
    action_cost = -0.01 if action != 0 else 0.0

    total_reward = shaped_progress + success_bonus + speed_penalty + action_cost

    components = {
        "progress": progress,
        "gate_angle": gate_angle,
        "contact_factor": contact_factor,
        "shaped_progress": shaped_progress,
        "speed_penalty": speed_penalty,
        "success_bonus": success_bonus,
        "action_cost": action_cost
    }
    return float(total_reward), components