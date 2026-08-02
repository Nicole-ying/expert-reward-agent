def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    v4: replaced hard-threshold soft_landing with proximity * speed_bonus (1/(1+speed))
        to give continuous slowdown rewards and eliminate multiplicative collapse.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05          # reduced; new form activates more often
    ANGLE_PENALTY_WEIGHT = 0.01
    ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.02

    PROXIMITY_THRESHOLD = 0.5      # distance within which we encourage slowing down
    ANGULAR_VELOCITY_THRESHOLD = 0.5

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive (continuous slowdown) ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    speed = (x_v_n ** 2 + y_v_n ** 2) ** 0.5
    speed_bonus = 1.0 / (1.0 + speed)   # smooth: 1 at rest, ~0.5 at speed=1, >0 always
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) angular velocity hinge penalty ----------
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