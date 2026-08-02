def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    v6: transform sparse success_bonus into continuous dense factor to restore training stability.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05
    ANGLE_PENALTY_WEIGHT = 0.01
    CONTACT_WEIGHT = 0.1
    PROXIMITY_THRESHOLD = 0.5
    SUCCESS_DIST_THRESHOLD = 0.3
    SUCCESS_SPEED_THRESHOLD = 0.3
    SUCCESS_ANGLE_THRESHOLD = 0.2
    SUCCESS_SCALE = 1.0          # was 50.0 for sparse bonus; now continuous

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
    speed_bonus = 1.0 / (1.0 + speed)
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) contact stability reward ----------
    contact_flag = max(left_n, right_n)
    angle_bonus = 1.0 / (1.0 + abs(angle_n))
    contact_stability = (
        CONTACT_WEIGHT * proximity * contact_flag * speed_bonus * angle_bonus
    )

    # ---------- 5) success bonus (dense continuous factor) ----------
    proximity_factor = max(0.0, 1.0 - R_next / SUCCESS_DIST_THRESHOLD)
    speed_factor = max(0.0, 1.0 - speed / SUCCESS_SPEED_THRESHOLD)
    angle_factor = max(0.0, 1.0 - abs(angle_n) / SUCCESS_ANGLE_THRESHOLD)
    contact_factor = max(left_n, right_n)
    success_factor = (proximity_factor * speed_factor *
                      angle_factor * contact_factor)
    success_bonus = SUCCESS_SCALE * success_factor

    # ---------- aggregate ----------
    total_reward = (
        progress_reward + soft_landing + angle_penalty +
        contact_stability + success_bonus
    )

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "contact_stability": contact_stability,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components