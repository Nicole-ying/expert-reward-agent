```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward: added fuel efficiency penalty for engine usage.
    """
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    both_contacts = (left_contact > 0.5) and (right_contact > 0.5)

    # progress toward target
    kp = 1.0
    w_progress = 1.0
    progress_reward = w_progress * (1.0 / (1.0 + kp * dist))

    # speed penalty when close to pad
    threshold_speed = 2.0
    w_speed = 0.5
    if dist < threshold_speed:
        landing_speed_penalty = -w_speed * (abs(x_vel) + abs(y_vel))
    else:
        landing_speed_penalty = 0.0

    # dual-contact reward
    w_contact = 2.0
    kc = 1.0
    if both_contacts:
        contact_reward = w_contact * (1.0 / (1.0 + kc * dist))
    else:
        contact_reward = 0.0

    # post-contact stillness penalty
    w_settle = 0.2
    if both_contacts:
        settlement_penalty = -w_settle * (abs(x_vel) + abs(y_vel) + abs(angular_vel) + abs(body_angle))
    else:
        settlement_penalty = 0.0

    # fuel efficiency penalty: discourage unnecessary engine use
    w_fuel = 0.05
    if action in (1, 2, 3):  # left_orientation_engine, main_engine, right_orientation_engine
        fuel_penalty = -w_fuel
    else:
        fuel_penalty = 0.0

    total_reward = progress_reward + landing_speed_penalty + contact_reward + settlement_penalty + fuel_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_speed_penalty": landing_speed_penalty,
        "contact_reward": contact_reward,
        "settlement_penalty": settlement_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components
```