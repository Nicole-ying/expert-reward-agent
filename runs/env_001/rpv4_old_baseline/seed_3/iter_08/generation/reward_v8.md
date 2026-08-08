```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前/下一步状态
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact

    # 上一状态
    x_prev, y_prev = obs[0], obs[1]
    xv_prev, yv_prev = obs[2], obs[3]
    left_prev = obs[6]
    right_prev = obs[7]

    # 距离与速度
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5
    speed_norm = (xv ** 2 + yv ** 2) ** 0.5
    speed_norm_prev = (xv_prev ** 2 + yv_prev ** 2) ** 0.5

    # 1. 距离进度奖励：朝目标移动即给分
    progress = dist_prev - dist_next
    w_progress = 50.0

    # 2. 速度惩罚：接近目标时过快的速度会受罚
    proximity_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed_penalty = speed_norm * proximity_factor
    w_speed = 1.0

    # 3. 姿态正则化：保持竖直、避免旋转
    angle_cost = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.01

    # 4. 高度惩罚：鼓励下降到目标垫平面
    height_penalty = abs(y)
    w_height = 2.0

    # 5. 接触奖励：引导腿部接地
    single_contact = left_contact + right_contact  # 0/1/2
    contact_reward = 5.0 * single_contact + 10.0 * both_legs  # 双足共20，单足5

    # 6. 首次软着陆大额奖励（基于状态转移，一次性）
    prev_landed = (left_prev * right_prev) and (dist_prev < 0.3) and (speed_norm_prev < 0.5)
    curr_landed = both_legs and (dist_next < 0.3) and (speed_norm < 0.5)
    landing_bonus = 200.0 if (curr_landed and not prev_landed) else 0.0

    # 7. 引擎使用微罚
    engine_penalty = 0.001 if action != 0 else 0.0

    total_reward = (
        w_progress * progress
        - w_speed * speed_penalty
        - w_ori * angle_cost
        - w_height * height_penalty
        + contact_reward
        + landing_bonus
        - engine_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty': -w_speed * speed_penalty,
        'orientation': -w_ori * angle_cost,
        'height_penalty': -w_height * height_penalty,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'engine_penalty': -engine_penalty,
    }

    return float(total_reward), components
```