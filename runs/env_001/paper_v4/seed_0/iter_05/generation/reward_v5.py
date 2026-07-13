def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract previous state
    prev_x = obs[0]
    prev_y = obs[1]
    prev_xv = obs[2]
    prev_yv = obs[3]
    prev_ang = obs[4]
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Extract next state
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_xv = next_obs[2]
    next_yv = next_obs[3]
    next_ang = next_obs[4]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    # 1. Progress delta: how much closer to the target center
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress_delta = prev_dist - next_dist  # positive = approaching, negative = receding

    # 2. Approach improvement: change in landing-quality potential
    def landing_quality(y, lc, rc, xv, yv, ang, threshold=0.5):
        if y < threshold:
            contact = lc + rc
            total_speed = abs(xv) + abs(yv)
            speed_ok = max(0.0, 1.0 - total_speed / 2.0)
            angle_ok = max(0.0, 1.0 - abs(ang) / 0.5)
            return contact * speed_ok * angle_ok
        return 0.0

    prev_quality = landing_quality(prev_y, prev_lc, prev_rc, prev_xv, prev_yv, prev_ang)
    next_quality = landing_quality(next_y, next_lc, next_rc, next_xv, next_yv, next_ang)
    approach_improvement = 2.0 * (next_quality - prev_quality)

    # 3. Contact event: reward first moment both legs touch ground
    prev_contacts = prev_lc + prev_rc
    next_contacts = next_lc + next_rc
    contact_event = 2.0 if (next_contacts == 2 and prev_contacts < 2) else 0.0

    # 4. Fuel efficiency
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = progress_delta + approach_improvement + contact_event + fuel_penalty

    components = {
        "progress_delta": progress_delta,
        "approach_improvement": approach_improvement,
        "contact_event": contact_event,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)