```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- unpack observations ----
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    x_vel_new, y_vel_new = next_obs[2], next_obs[3]
    body_angle_new = next_obs[4]
    ang_vel_new = next_obs[5]
    left_contact_new = next_obs[6]
    right_contact_new = next_obs[7]

    # 1. Approach reward – milder to avoid reckless speeding
    dist_old = (x_old**2 + y_old**2) ** 0.5
    dist_new = (x_new**2 + y_new**2) ** 0.5
    approach_reward = 10.0 * (dist_old - dist_new)

    # 2. Altitude reward – encourage reaching the landing pad height (y≈0)
    altitude_reward = 10.0 / (1.0 + 5.0 * abs(y_new))

    # 3. Velocity penalty – baseline penalty, amplified near ground to force a soft landing
    speed = abs(x_vel_new) + abs(y_vel_new)
    near_ground_scale = 3.0 / (abs(y_new) + 0.1)
    vel_penalty = -speed * (1.0 + near_ground_scale)

    # 4. Stability penalty – moderate tilt and angular velocity cost
    stability_penalty = -10.0 * (body_angle_new**2) - 1.0 * (ang_vel_new**2)

    # 5. Thrust cost – small penalty for any engine use to promote fuel efficiency
    thrust_cost = -0.03 if action != 0 else 0.0

    # 6. Soft landing bonus – large reward for gentle two-leg touchdown
    contact_both = 1.0 if (left_contact_new > 0.5 and right_contact_new > 0.5) else 0.0
    if contact_both > 0.5:
        vel_sum = abs(x_vel_new) + abs(y_vel_new)
        angle_abs = abs(body_angle_new)
        vel_factor = 1.0 / (1.0 + 10.0 * vel_sum)
        angle_factor = 1.0 / (1.0 + 5.0 * angle_abs)
        landing_bonus = 500.0 * vel_factor * angle_factor
    else:
        landing_bonus = 0.0

    # ---- assemble ----
    total_reward = (approach_reward +
                    altitude_reward +
                    vel_penalty +
                    stability_penalty +
                    thrust_cost +
                    landing_bonus)
    components = {
        'approach_reward': approach_reward,
        'altitude_reward': altitude_reward,
        'vel_penalty': vel_penalty,
        'stability_penalty': stability_penalty,
        'thrust_cost': thrust_cost,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```