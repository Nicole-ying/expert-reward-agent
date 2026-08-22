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

    # 1. 进度奖励：每一步缩小的距离
    progress = dist_prev - dist_next
    w_progress = 2.0

    # 2. 全局速度惩罚，鼓励全程保持低速，便于控制
    speed_penalty_global = speed_norm ** 2
    w_speed_global = 0.1

    # 3. 近端速度惩罚，在接近目标时强烈要求减速
    near_gate = 1.0 / (1.0 + dist_next)
    speed_penalty_near = speed_norm ** 2
    w_speed_near = 2.0

    # 4. 垂直速度惩罚，接近地面时防止重着陆
    ground_prox = 1.0 / (1.0 + abs(y) + 1e-5)
    vert_speed_penalty = yv ** 2
    w_vert_speed = 0.5

    # 5. 着陆引导：接触奖励 + 接近/低速额外奖励
    dist_gate = 1.0 / (1.0 + 10.0 * dist_next)
    speed_gate = 1.0 / (1.0 + 10.0 * speed_norm)
    contact_base = 5.0 * both_legs * (1.0 / (1.0 + dist_next))  # 越近接触越有价值
    landing_bonus = 200.0 * dist_gate * speed_gate * both_legs
    landing_reward = contact_base + landing_bonus

    # 6. 姿态正则化（保持平稳，避免旋转）
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.001

    total_reward = (
        w_progress * progress
        - w_speed_global * speed_penalty_global
        - w_speed_near * near_gate * speed_penalty_near
        - w_vert_speed * ground_prox * vert_speed_penalty
        + landing_reward
        - w_ori * angle_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty_global': -w_speed_global * speed_penalty_global,
        'speed_penalty_near': -w_speed_near * near_gate * speed_penalty_near,
        'vert_speed_penalty': -w_vert_speed * ground_prox * vert_speed_penalty,
        'landing': landing_reward,
        'orientation': -w_ori * angle_penalty,
    }

    return float(total_reward), components
```