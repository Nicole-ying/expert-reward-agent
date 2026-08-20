def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Unpack previous state for distance change
    prev_px, prev_py = obs[0], obs[1]

    curr_dist = (px**2 + py**2)**0.5
    prev_dist = (prev_px**2 + prev_py**2)**0.5

    # 1. Progress toward center
    dist_reward = 0.5 * (prev_dist - curr_dist)

    # 2. Small penalty for being far away (shapes final position)
    distance_penalty = -0.1 * curr_dist

    # 3. Keep upright
    angle_penalty = -0.5 * (angle**2)

    # 4. Encourage low speed
    vel_penalty = -0.2 * (vx**2 + vy**2)

    # 5. Penalize excessive rotation
    angvel_penalty = -0.02 * (ang_vel**2)

    # 6. Soft contact incentive (magnitude drastically reduced)
    contact_reward = 1.0 * left_contact + 1.0 * right_contact + 2.0 * (left_contact * right_contact)

    # 7. Success bonus for stable touchdown near center
    success_bonus = 2.0 if (left_contact and right_contact and curr_dist < 0.5) else 0.0

    total = (dist_reward + distance_penalty + angle_penalty +
             vel_penalty + angvel_penalty + contact_reward + success_bonus)

    components = {
        "dist_reward": dist_reward,
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }
    return float(total), components