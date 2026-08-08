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

    # 1. 进度奖励：鼓励向目标移动
    progress = dist_prev - dist_next
    w_progress = 15.0

    # 2. 全局速度惩罚：鼓励整体减速
    speed_penalty = speed_norm
    w_speed = 0.1

    # 3. 姿态正则化：保持机体平稳
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.01

    # 4. 微弱接近引导：只用于初期的方向指示，避免 hover
    proximity = 0.5 / (1.0 + 20.0 * dist_next ** 2)

    # 5. 软着陆奖励：仅在双足接触、靠近目标且低速时给予
    soft_landing = 10.0 * both_legs / (1.0 + 50.0 * dist_next ** 2) / (1.0 + 10.0 * speed_norm ** 2)

    # 6. 接触鼓励：在接近目标时双足接触且速度不太高时给予阶段性奖励
    near_factor = 1.0 / (1.0 + 100.0 * dist_next ** 2)
    contact_enc = 0.0
    if both_legs and speed_norm < 2.0:
        contact_enc = near_factor * 10.0 * (1.0 - speed_norm / 2.0)

    # 7. 引擎惩罚：鼓励节约燃料（次要目标）
    engine_penalty = 0.05 if action != 0 else 0.0

    total_reward = (
        w_progress * progress
        - w_speed * speed_penalty
        - w_ori * angle_penalty
        + proximity
        + soft_landing
        + contact_enc
        - engine_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty': -w_speed * speed_penalty,
        'orientation': -w_ori * angle_penalty,
        'proximity': proximity,
        'soft_landing': soft_landing,
        'contact_encouragement': contact_enc,
        'engine_penalty': -engine_penalty,
    }

    return float(total_reward), components
```
