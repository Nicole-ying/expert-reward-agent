def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_left = obs[6]
    prev_right = obs[7]

    curr_dist = (px**2 + py**2)**0.5

    # Core drive toward origin (proven in iter 3, ~-130/ep)
    distance_penalty = -0.5 * curr_dist

    # One-time contact onset bonus — fires only on transition 0→1
    left_onset = 1.0 if (left_contact > 0.5 and prev_left < 0.5) else 0.0
    right_onset = 1.0 if (right_contact > 0.5 and prev_right < 0.5) else 0.0
    contact_reward = 40.0 * left_onset + 40.0 * right_onset + 80.0 * (left_onset * right_onset)

    # Per-step stable-landing bonus (small, only during successful touchdown)
    success_bonus = 2.0 if (left_contact and right_contact and curr_dist < 0.3 and abs(angle) < 0.2) else 0.0

    # Keep upright
    angle_penalty = -0.5 * (angle ** 2)

    # Moderate speed penalty
    vel_penalty = -0.1 * (vx ** 2 + vy ** 2)

    # Smooth rotation
    angvel_penalty = -0.05 * (ang_vel ** 2)

    total = (distance_penalty + contact_reward + success_bonus +
             angle_penalty + vel_penalty + angvel_penalty)

    components = {
        "distance_penalty": distance_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "angvel_penalty": angvel_penalty
    }
    return float(total), components