# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact

    # 距离与速度
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5
    speed_norm = (xv ** 2 + yv ** 2) ** 0.5

    # 1. 进度奖励：向目标移动
    progress = dist_prev - dist_next
    w_progress = 15.0

    # 2. 速度惩罚（仅在接近目标时显著）
    proximity_factor = 1.0 / (1.0 + 10.0 * dist_next)  # 0~1，越近越大
    speed_penalty = speed_norm * proximity_factor
    w_speed = 0.5

    # 3. 姿态正则化
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.01

    # 4. 接近引导
    proximity = 0.5 / (1.0 + 20.0 * dist_next ** 2)

    # 5. 接触奖励（双足接地即给，鼓励着陆）
    contact_reward = 10.0 * both_legs

    # 6. 软着陆奖励（双足接地、靠近中心且低速）
    soft_landing = 0.0
    if both_legs and dist_next < 0.3 and speed_norm < 0.5:
        soft_landing = 30.0

    # 7. 引擎惩罚（极微弱，不干扰主任务）
    engine_penalty = 0.001 if action != 0 else 0.0

    total_reward = (
        w_progress * progress
        - w_speed * speed_penalty
        - w_ori * angle_penalty
        + proximity
        + contact_reward
        + soft_landing
        - engine_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty': -w_speed * speed_penalty,
        'orientation': -w_ori * angle_penalty,
        'proximity': proximity,
        'contact_reward': contact_reward,
        'soft_landing': soft_landing,
        'engine_penalty': -engine_penalty,
    }

    return float(total_reward), components
```
