def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense signal – progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : penalty for linear velocity (encourages slow approach)
      touchdown_reward    : dense reward for high-quality leg contact during soft landing
      soft_landing_bonus  : event-based – one-time +200 when a stable landing is first achieved
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty ----------
    vx = next_obs[2]
    vy = next_obs[3]
    speed_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---------- touchdown reward (leg contact quality) ----------
    # Only reward leg contact when the vehicle is stable (low speed, low tilt, near target)
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5   # at least one leg touching
    if leg_contact:
        dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
        speed = (vx ** 2 + vy ** 2) ** 0.5
        ang = abs(angle)
        # quality: 1.0 for each condition fully satisfied, linearly down to 0
        quality = max(0.0, 1.0 - dist / 0.3) * max(0.0, 1.0 - speed / 0.3) * max(0.0, 1.0 - ang / 0.2)
        touchdown_reward = 10.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- soft landing bonus (one-time success event) ----------
    def success(ob):
        leg_contact_s = (ob[6] + ob[7]) > 1.5           # both legs must touch
        pos_ok = abs(ob[0]) < 0.3                       # horizontal offset
        speed_ok = (ob[2]**2 + ob[3]**2)**0.5 < 0.3     # low linear speed
        angle_ok = abs(ob[4]) < 0.2                      # low tilt
        return leg_contact_s and pos_ok and speed_ok and angle_ok

    old_success = success(obs)
    new_success = success(next_obs)
    landing_event = (not old_success) and new_success
    soft_landing_bonus = 200.0 if landing_event else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + soft_landing_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "soft_landing_bonus": soft_landing_bonus,
    }
    return float(total_reward), components