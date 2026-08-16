def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress    : dense progress towards origin (target at (0,0))
      attitude_penalty      : penalty for tilt and angular velocity
      speed_penalty         : stage-gated penalty on linear velocity (higher near ground)
      first_touchdown_bonus : one-time reward for initial leg contact
      contact_reward        : per-step reward while both legs gently touch the ground
      success_bonus         : large terminal reward for successful landing (via info)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty (stage-gated) ----------
    y = next_obs[1]                   # vertical coordinate
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    if y < 0.5:                       # near ground – require stronger braking
        speed_penalty = -0.15 * speed_sq
    else:                             # in flight – allow faster descent
        speed_penalty = -0.01 * speed_sq

    # ---------- first touchdown bonus (event, not per-step) ----------
    old_leg_contact = (obs[6] + obs[7]) > 0.5
    new_leg_contact = (next_obs[6] + next_obs[7]) > 0.5
    first_touchdown = (not old_leg_contact) and new_leg_contact
    first_touchdown_bonus = 30.0 if first_touchdown else 0.0

    # ---------- contact reward (both legs down, gentle conditions) ----------
    leg_contact_stable = (next_obs[6] + next_obs[7]) > 1.5
    speed = (speed_sq) ** 0.5
    angle_mag = abs(angle)
    if leg_contact_stable and speed < 0.3 and angle_mag < 0.2:
        contact_reward = 2.0
    else:
        contact_reward = 0.0

    # ---------- success bonus (terminal, via info) ----------
    success = bool(info.get('success', False))
    success_bonus = 200.0 if success else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + first_touchdown_bonus
                    + contact_reward
                    + success_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "first_touchdown_bonus": first_touchdown_bonus,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components