```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于计算距离缩短）
    x = obs[0]
    y = obs[1]
    # 下一状态（用于评估动作后果）
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nleft = next_obs[6]
    nright = next_obs[7]

    # 距离缩短奖励：直接鼓励每一步靠近目标
    dist = (x * x + y * y) ** 0.5 + 1e-8
    next_dist = (nx * nx + ny * ny) ** 0.5 + 1e-8
    proximity_reward = 5.0 * (dist - next_dist)

    # 水平偏移惩罚：平方惩罚强约束，防止飞出视口
    x_penalty = -2.0 * (nx * nx)

    # 速度惩罚：柔和抑制过快运动
    speed_penalty = -0.3 * (nvx * nvx + nvy * nvy)

    # 姿态惩罚：鼓励水平
    angle_penalty = -0.5 * abs(nangle)

    # 引擎使用惩罚（节油）
    if action == 0:
        engine_penalty = 0.0
    elif action in (1, 3):
        engine_penalty = -0.15
    elif action == 2:
        engine_penalty = -0.4
    else:
        engine_penalty = 0.0

    # 固定时间惩罚
    time_penalty = -0.05

    # 着陆奖励
    any_contact = (nleft > 0.5 or nright > 0.5)
    full_contact = (nleft > 0.5 and nright > 0.5)
    near_target = (abs(nx) < 0.4 and ny < 0.3)

    if any_contact:
        contact_speed = (nvx * nvx + nvy * nvy) ** 0.5
        contact_speed_penalty = -1.0 * contact_speed
        contact_angle_penalty = -1.0 * abs(nangle)

        if full_contact and near_target:
            if contact_speed < 0.2 and abs(nangle) < 0.2:
                landing_bonus = 600.0 + contact_speed_penalty + contact_angle_penalty
            else:
                landing_bonus = 200.0 + contact_speed_penalty + contact_angle_penalty
        else:
            landing_bonus = contact_speed_penalty + contact_angle_penalty
    else:
        landing_bonus = 0.0

    total_reward = (proximity_reward + x_penalty + speed_penalty + angle_penalty +
                    engine_penalty + time_penalty + landing_bonus)

    components = {
        'proximity_reward': proximity_reward,
        'x_penalty': x_penalty,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'engine_penalty': engine_penalty,
        'time_penalty': time_penalty,
        'landing_bonus': landing_bonus,
    }

    return float(total_reward), components
```