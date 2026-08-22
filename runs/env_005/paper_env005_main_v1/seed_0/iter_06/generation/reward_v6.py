def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Materially different reward design (v2) — adds height deviation penalty
    to prevent body collapse, which caused 19/20 early terminations in v1.
    All v1 components and weights are preserved unchanged.
    """
    # next_obs indices
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion components: assumed order (w, x, y, z)
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Body height (vertical position)
    body_z = next_obs[0]

    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights — v1 weights preserved
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05

    # New height deviation penalty
    # Healthy range: [0.2, 1.0], center = 0.6, half-range = 0.4
    # At body_z=0.2 or 1.0, penalty = -1.0; at center 0.6, penalty = 0
    w_height = 1.0
    height_deviation = (body_z - 0.6) / 0.4
    height_deviation_penalty = -w_height * (height_deviation ** 2)

    # Components (v1 preserved)
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_deviation_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_deviation_penalty": height_deviation_penalty,
    }
    return float(total_reward), components