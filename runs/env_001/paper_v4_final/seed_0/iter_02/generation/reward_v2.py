def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    Drives the agent to reach the target pad and settle gently.
    v2: added angular velocity hinge penalty to prevent attitude crash.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.2
    ANGLE_PENALTY_WEIGHT = 0.01
    ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.02

    PROXIMITY_THRESHOLD = 0.5    # distance to start shaping soft landing
    ANGLE_THRESHOLD = 0.5        # rad
    VELOCITY_THRESHOLD = 0.5     # sum of absolute linear velocities
    ANGULAR_VELOCITY_THRESHOLD = 0.5  # rad/s, safety boundary for attitude control

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)   # positive when getting closer

    # ---------- 2) soft landing incentive ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / ANGLE_THRESHOLD)
    vel_ok = max(0.0, 1.0 - (abs(x_v_n) + abs(y_v_n)) / VELOCITY_THRESHOLD)
    contact_ok = left_n * right_n   # both support feet on the ground

    soft_landing = LANDING_WEIGHT * proximity * angle_ok * vel_ok * contact_ok

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) angular velocity hinge penalty ----------
    # Penalize excessive angular velocity that leads to attitude crash.
    # Hinge form: only activates when |angvel| exceeds threshold.
    # This is a soft gate precursor — not a full gate, just a disincentive.
    angular_velocity_penalty = (
        -ANGULAR_VELOCITY_PENALTY_WEIGHT
        * max(0.0, abs(angvel_n) - ANGULAR_VELOCITY_THRESHOLD)
    )

    # ---------- aggregate ----------
    total_reward = progress_reward + soft_landing + angle_penalty + angular_velocity_penalty

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return float(total_reward), components