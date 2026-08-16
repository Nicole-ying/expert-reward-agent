def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense - progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : stage‑gated penalty on linear velocity (loose when high, strict near ground)
      touchdown_reward    : additive quality reward for any leg contact
      landing_quality_reward : per‑step reward while fully stable on the pad (replaces one‑time bonus)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty (stage‑gated) ----------
    y = next_obs[1]                   # vertical coordinate, decreasing towards ground
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    if y < 0.5:                       # near ground – strict braking
        speed_penalty = -0.1 * speed_sq
    else:                             # in flight – allow faster descent
        speed_penalty = -0.01 * speed_sq

    # ---------- touchdown reward (leg contact, additive quality) ----------
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5    # at least one leg touching
    if leg_contact:
        dist = new_dist
        speed = speed_sq ** 0.5
        ang = abs(angle)
        # additive factors: each independently rewards closeness to target conditions
        q_dist  = max(0.0, 1.0 - dist / 0.3)
        q_speed = max(0.0, 1.0 - speed / 0.3)
        q_angle = max(0.0, 1.0 - ang / 0.2)
        quality = q_dist + q_speed + q_angle   # range 0..3
        touchdown_reward = 3.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- landing quality (per‑step stable‑landing reward) ----------
    leg_contact_stable = (next_obs[6] + next_obs[7]) > 1.5      # both legs firmly
    pos_ok   = abs(next_obs[0]) < 0.3
    speed_ok = speed_sq ** 0.5 < 0.3
    angle_ok = abs(angle) < 0.2
    if leg_contact_stable and pos_ok and speed_ok and angle_ok:
        landing_quality_reward = 10.0
    else:
        landing_quality_reward = 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + landing_quality_reward)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "landing_quality_reward": landing_quality_reward,
    }
    return float(total_reward), components