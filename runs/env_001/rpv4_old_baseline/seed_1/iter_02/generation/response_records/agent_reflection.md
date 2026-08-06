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

    # 1. Approach reward – smaller coefficient to avoid reckless speeding
    dist_old = (x_old**2 + y_old**2) ** 0.5
    dist_new = (x_new**2 + y_new**2) ** 0.5
    approach_reward = 30.0 * (dist_old - dist_new)

    # 2. Linear velocity penalty – encourages slow, controlled motion
    vel_penalty = -1.0 * (abs(x_vel_new) + abs(y_vel_new))

    # 3. Attitude and angular-velocity penalty – stronger to keep the vehicle upright
    stability_penalty = -20.0 * (body_angle_new**2) - 2.0 * (ang_vel_new**2)

    # 4. Thrust cost – penalise any engine use to promote fuel efficiency
    thrust_cost = -0.03 if action != 0 else 0.0

    # 5. Unbalanced contact penalty – discourages landing on only one leg
    sum_contacts = left_contact_new + right_contact_new
    unbalanced_penalty = -50.0 if (sum_contacts > 0.4 and sum_contacts < 1.6) else 0.0

    # 6. Soft landing bonus – large reward for gentle two‑leg touchdown
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
                    vel_penalty +
                    stability_penalty +
                    thrust_cost +
                    unbalanced_penalty +
                    landing_bonus)
    components = {
        'approach_reward': approach_reward,
        'vel_penalty': vel_penalty,
        'stability_penalty': stability_penalty,
        'thrust_cost': thrust_cost,
        'unbalanced_penalty': unbalanced_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```
