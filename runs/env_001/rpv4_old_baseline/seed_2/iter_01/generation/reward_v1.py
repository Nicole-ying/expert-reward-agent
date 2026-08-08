def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables from next_obs (post-action state)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Proximity reward (main learning signal)
    # Drive the agent toward the target platform (x=0, y=0)
    dist_sq = x * x + y * y
    proximity_reward = -0.5 * dist_sq

    # 2. Velocity penalty (soft landing constraint)
    # Penalise excessive horizontal and vertical velocity
    v_thresh = 0.2
    v_penalty = 0.0
    if abs(vx) > v_thresh:
        v_penalty += abs(vx) - v_thresh
    if abs(vy) > v_thresh:
        v_penalty += abs(vy) - v_thresh
    velocity_penalty = -1.0 * v_penalty

    # 3. Body angle penalty (stability constraint)
    # Penalise tilting away from upright
    angle_thresh = 0.1  # radians
    if abs(body_angle) > angle_thresh:
        angle_penalty = -1.0 * (abs(body_angle) - angle_thresh)
    else:
        angle_penalty = 0.0

    # 4. Landing bonus (task completion proxy)
    # Strong bonus when both legs touch with low speed, angle, and angular velocity
    k_v = 5.0
    k_angle = 10.0
    k_angvel = 5.0

    vx_factor = 1.0 / (1.0 + k_v * abs(vx))
    vy_factor = 1.0 / (1.0 + k_v * abs(vy))
    angle_factor = 1.0 / (1.0 + k_angle * abs(body_angle))
    angvel_factor = 1.0 / (1.0 + k_angvel * abs(angvel))
    contact_factor = left_contact * right_contact  # only if both feet touch

    landing_bonus = 20.0 * contact_factor * vx_factor * vy_factor * angle_factor * angvel_factor

    total = proximity_reward + velocity_penalty + angle_penalty + landing_bonus
    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "landing_bonus": landing_bonus
    }
    return total, components