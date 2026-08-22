def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state
    px, py, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left_contact, right_contact = next_obs[6], next_obs[7]

    # ---- New core signal: distance penalty ----
    # Agent is heavily pushed to minimise its distance to the centre (0,0)
    curr_dist = (px ** 2 + py ** 2) ** 0.5
    distance_penalty = -0.5 * curr_dist

    # ---- Attitude (keep upright) ----
    angle_penalty = -0.5 * (angle ** 2)

    # ---- Rotation smoothness ----
    angvel_penalty = -0.1 * (ang_vel ** 2)

    # ---- Linear speed: now a gentle penalty (was 0.2, dwarfed everything) ----
    vel_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---- Landing contact: much stronger incentive ----
    contact_reward = 10.0 * left_contact + 10.0 * right_contact + 30.0 * left_contact * right_contact

    total = distance_penalty + angle_penalty + angvel_penalty + vel_penalty + contact_reward
    components = {
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vel_penalty": vel_penalty,
        "contact_reward": contact_reward
    }
    return float(total), components