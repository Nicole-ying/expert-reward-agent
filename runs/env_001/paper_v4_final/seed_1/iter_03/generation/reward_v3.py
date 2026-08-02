def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ------------------  Main progress signal: distance reduction  ------------------
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = (dist - next_dist)

    # -----------  Landing incentive with soft contact gate  -----------
    # Gate: 0.1 when no leg contact, 1.0 when at least one leg in contact
    # This makes landing ~10x more rewarding than hovering nearby
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    contact_gate = 0.1 + 0.9 * leg_contact

    # Continuous proximity bonus, gated by actual contact
    w_landing = 0.5
    landing_incentive = contact_gate * w_landing / (1.0 + next_dist * 5.0)

    # -------------------  Health constraint: body angle (tightened)  -------------------
    # Tightened safe_angle from 0.5 -> 0.3 rad (~17 degrees)
    # Makes the penalty actually engage before extreme angles
    w_angle = 0.5
    safe_angle = 0.3
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # -------------------  Total reward  -------------------
    total_reward = w_progress * progress + landing_incentive + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components