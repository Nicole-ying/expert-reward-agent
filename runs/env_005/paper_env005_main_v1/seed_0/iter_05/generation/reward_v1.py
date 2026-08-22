def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Materially different reward design (v1) — no height-dependent terms.
    Focuses purely on forward motion, upright orientation, and suppressing
    lateral drift and angular rates, without any explicit height penalty.
    """
    # next_obs indices:
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion components: assumed order (w, x, y, z)
    # Body up direction z-component in world frame
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]  # not used for body_up_z but included for completeness
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights — chosen to keep forward reward dominant
    w_fwd   = 1.0
    w_orient = 0.3
    w_lat   = 0.2
    w_ang   = 0.05

    # Components
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = forward_reward + orientation_reward + lateral_penalty + angular_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty
    }
    return float(total_reward), components