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

    # 1. 径向速度奖励：始终鼓励向目标移动
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y
    radial_reward = 2.0 * radial_vel

    # 2. 水平漂移惩罚：抑制侧向速度
    vx_penalty = -0.3 * abs(vx)

    # 3. 垂直速度引导：根据高度把vy维持在一个安全区间
    if y > 0.5:
        vy_low, vy_high = -0.5, -0.1
    elif y > 0.1:
        vy_low, vy_high = -0.3, -0.05
    else:
        vy_low, vy_high = -0.15, -0.02

    if vy < vy_low:
        vy_penalty = -0.5 * (vy_low - vy)   # 下降太快
    elif vy > vy_high:
        vy_penalty = -0.5 * (vy - vy_high)  # 上升或太慢
    else:
        vy_penalty = 0.0

    # 4. 姿态惩罚：轻微惩罚倾斜
    angle_penalty = -0.5 * abs(angle)

    # 5. 引擎使用惩罚：鼓励节油
    if action == 0:
        engine_penalty = 0.0
    elif action in (1, 3):
        engine_penalty = -0.15
    elif action == 2:
        engine_penalty = -0.4
    else:
        engine_penalty = 0.0

    # 6. 水平位置惩罚：防止飞出视口
    x_penalty = -0.2 * abs(x)

    # 7. 下降进度奖励：适度鼓励高度降低
    descent_reward = 0.5 * max(0.0, obs[1] - next_obs[1])

    # 8. 着陆奖励（接触时触发）
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x) < 0.4 and y < 0.3)

    if any_contact:
        speed = (vx * vx + vy * vy) ** 0.5
        contact_penalty = -2.0 * speed - 2.0 * abs(angle)
        if full_contact and near_target:
            if speed < 0.3 and abs(angle) < 0.3:
                landing_bonus = 400.0 + contact_penalty
            else:
                landing_bonus = 150.0 + contact_penalty
        else:
            landing_bonus = contact_penalty
    else:
        landing_bonus = 0.0

    # 9. 时间惩罚：鼓励尽快完成任务
    time_penalty = -0.05

    total_reward = (radial_reward + vx_penalty + vy_penalty + angle_penalty +
                    engine_penalty + x_penalty + descent_reward +
                    landing_bonus + time_penalty)

    components = {
        'radial_reward': radial_reward,
        'vx_penalty': vx_penalty,
        'vy_penalty': vy_penalty,
        'angle_penalty': angle_penalty,
        'engine_penalty': engine_penalty,
        'x_penalty': x_penalty,
        'descent_reward': descent_reward,
        'landing_bonus': landing_bonus,
        'time_penalty': time_penalty,
    }

    return float(total_reward), components
```