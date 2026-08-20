def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward function for the soft landing / docking task.

    Roles:
      - approach_target: continuous distance reward (bounded, positive)
      - safe_landing: speed penalty near pad + contact bonus for both legs
      - settlement_stabilization: penalty for velocity/angular movement after contact
    """
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # helpers
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    both_contacts = (left_contact > 0.5) and (right_contact > 0.5)

    # component A: approach_target – positive bounded distance reward
    # using 1/(1 + kp*dist) → max 1.0 when exactly on target
    kp = 1.0
    w_progress = 1.0
    progress_reward = w_progress * (1.0 / (1.0 + kp * dist))

    # component B: safe_landing – speed penalty when close to pad, contact bonus
    threshold_speed = 2.0        # apply speed penalty only within this distance
    w_speed = 0.5                # penalty weight for |vx|+|vy|
    if dist < threshold_speed:
        landing_speed_penalty = -w_speed * (abs(x_vel) + abs(y_vel))
    else:
        landing_speed_penalty = 0.0

    # contact bonus: both legs touching, higher when near center
    w_contact = 2.0
    kc = 1.0
    if both_contacts:
        contact_reward = w_contact * (1.0 / (1.0 + kc * dist))
    else:
        contact_reward = 0.0

    # component C: settlement_stabilization – after contact, encourage full stillness
    w_settle = 0.2
    if both_contacts:
        settlement_penalty = -w_settle * (
            abs(x_vel) + abs(y_vel) + abs(angular_vel) + abs(body_angle)
        )
    else:
        settlement_penalty = 0.0

    total_reward = progress_reward + landing_speed_penalty + contact_reward + settlement_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_speed_penalty": landing_speed_penalty,
        "contact_reward": contact_reward,
        "settlement_penalty": settlement_penalty,
    }

    return float(total_reward), components