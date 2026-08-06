```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    dist = (x * x + y * y) ** 0.5 + 1e-8

    # 1. Radial velocity reward: positive when moving toward target
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y
    radial_reward = 3.0 * radial_vel

    # 2. Velocity shaping: penalise horizontal drift and encourage controlled descent
    if y > 0.3:
        desired_vy = -0.4
    elif y > 0.1:
        desired_vy = -0.25
    else:
        desired_vy = -0.05
    vy_error = (vy - desired_vy) ** 2
    vx_error = vx ** 2
    vel_penalty = -0.8 * vx_error - 0.6 * vy_error

    # 3. Attitude penalty: stay upright
    angle_penalty = -2.0 * abs(angle)

    # 4. Engine usage penalty
    if action == 0:
        engine_penalty = 0.0
    elif action in (1, 3):   # orientation engines
        engine_penalty = -0.2
    elif action == 2:        # main engine
        engine_penalty = -0.5
    else:
        engine_penalty = 0.0

    # 5. Small proximity bonus, only when descending
    if y < 0.5 and dist < 1.5 and vy < -0.1:
        proximity_bonus = 0.5 * (1.5 - dist) * max(0.0, 0.5 - y)
    else:
        proximity_bonus = 0.0

    # 6. Descent progress: reward altitude reduction
    descent_reward = 2.0 * max(0.0, obs[1] - next_obs[1])

    # 7. Landing and contact bonuses / penalties
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x) < 0.4 and y < 0.3)

    if any_contact:
        speed = (vx * vx + vy * vy) ** 0.5
        # Severe crash penalty
        if speed > 1.2 or abs(angle) > 0.8:
            landing_bonus = -100.0
        elif full_contact and near_target:
            if speed < 0.6 and abs(angle) < 0.5:
                landing_bonus = 500.0   # perfect landing
            else:
                landing_bonus = 200.0   # acceptable landing near target
        elif full_contact and not near_target:
            landing_bonus = -50.0       # landed in wrong place
        else:  # partial contact (single leg)
            landing_bonus = 5.0 if near_target else -10.0
    else:
        landing_bonus = 0.0

    # 8. Per-step time penalty
    time_penalty = -0.05

    total_reward = (radial_reward + vel_penalty + angle_penalty +
                    engine_penalty + proximity_bonus + descent_reward +
                    landing_bonus + time_penalty)

    components = {
        'radial_reward': radial_reward,
        'vel_penalty': vel_penalty,
        'angle_penalty': angle_penalty,
        'engine_penalty': engine_penalty,
        'proximity_bonus': proximity_bonus,
        'descent_reward': descent_reward,
        'landing_bonus': landing_bonus,
        'time_penalty': time_penalty,
    }

    return float(total_reward), components
```