def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyper-parameters (v1) ----------
    PROGRESS_WEIGHT = 2.0          # weight for distance reduction
    SOFT_LAND_WEIGHT = 0.5         # weight for speed penalty near pad
    LANDING_THRESHOLD = 1.0        # distance below which speed penalty activates
    ANGLE_WEIGHT = 1.0             # weight for body angle error
    ANGVEL_WEIGHT = 0.5            # weight for angular velocity penalty

    # ---------- compute current and next distance to target pad ----------
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # ---------- main learning signal: approach progress ----------
    progress = dist_curr - dist_next   # positive when moving closer
    reward_progress = PROGRESS_WEIGHT * progress

    # ---------- stability constraint 1: soft-landing regularization ----------
    speed = abs(next_obs[2]) + abs(next_obs[3])
    landing_gate = max(0.0, 1.0 - dist_next / LANDING_THRESHOLD)
    soft_land_penalty = -SOFT_LAND_WEIGHT * speed * landing_gate

    # ---------- stability constraint 2: orientation stability ----------
    angle = next_obs[4]
    angvel = next_obs[5]
    orientation_penalty = -ANGLE_WEIGHT * (angle**2) - ANGVEL_WEIGHT * (angvel**2)

    # ---------- total reward ----------
    total_reward = reward_progress + soft_land_penalty + orientation_penalty

    components = {
        "progress": reward_progress,
        "soft_land_penalty": soft_land_penalty,
        "orientation_penalty": orientation_penalty
    }

    return float(total_reward), components