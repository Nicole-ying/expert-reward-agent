# Response Record

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

    # 1. Progress reward: only reward moving closer to the landing target (x=0,y=0)
    dist_old = (x_old**2 + y_old**2) ** 0.5
    dist_new = (x_new**2 + y_new**2) ** 0.5
    progress_reward = 5.0 * max(0.0, dist_old - dist_new)

    # 2. Position attractor: encourage staying near the pad center
    pos_attract = 2.0 / (1.0 + abs(x_new) + abs(y_new))

    # 3. Velocity penalty: base cost, amplified when close to ground to force a soft landing
    speed = abs(x_vel_new) + abs(y_vel_new)
    near_ground_factor = 2.0 / (abs(y_new) + 0.2)
    vel_penalty = -speed * (0.5 + near_ground_factor)

    # 4. Stability penalty: moderate cost for tilt and angular velocity
    stability_penalty = -2.0 * (body_angle_new**2) - 0.5 * (ang_vel_new**2)

    # 5. Thrust cost: small penalty for using engines
    thrust_cost = -0.05 if action != 0 else 0.0

    # 6. Contact reward: reward any leg contact to encourage touchdown attempts
    contact_reward = 2.0 * (left_contact_new + right_contact_new)

    # 7. Soft landing bonus: large reward when both legs touch with low speed and upright body
    contact_both = 1.0 if (left_contact_new > 0.5 and right_contact_new > 0.5) else 0.0
    if contact_both > 0.5:
        vel_sum = abs(x_vel_new) + abs(y_vel_new)
        angle_abs = abs(body_angle_new)
        vel_factor = 1.0 / (1.0 + 10.0 * vel_sum)
        angle_factor = 1.0 / (1.0 + 5.0 * angle_abs)
        landing_bonus = 500.0 * vel_factor * angle_factor
    else:
        landing_bonus = 0.0

    # 8. Step cost: small constant penalty to discourage excessive hovering
    step_cost = -0.1

    # ---- assemble ----
    total_reward = (progress_reward +
                    pos_attract +
                    vel_penalty +
                    stability_penalty +
                    thrust_cost +
                    contact_reward +
                    landing_bonus +
                    step_cost)
    components = {
        'progress_reward': progress_reward,
        'pos_attract': pos_attract,
        'vel_penalty': vel_penalty,
        'stability_penalty': stability_penalty,
        'thrust_cost': thrust_cost,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'step_cost': step_cost
    }
    return float(total_reward), components
```
