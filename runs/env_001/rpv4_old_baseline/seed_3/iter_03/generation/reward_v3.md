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

    # 距离计算
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5
    speed_norm = (xv ** 2 + yv ** 2) ** 0.5

    # 1. 靠近目标进度
    progress = dist_prev - dist_next
    w_progress = 0.5

    # 2. 速度方向奖励：速度在指向原点方向的分量
    dot = x * xv + y * yv
    approach_vel = -dot / (dist_next + 1e-5)
    w_approach = 0.3

    # 3. 近端减速引导（只在接近目标时惩罚速度大小）
    near_gate = 1.0 / (1.0 + dist_next)
    speed_penalty = speed_norm ** 2
    w_speed_near = 0.1

    # 4. 着陆成功大奖励（软门控，仅当非常接近、速度极低、双脚接触时生效）
    landing_prox = 1.0 / (1.0 + 100.0 * dist_next ** 2)
    landing_low_speed = 1.0 / (1.0 + 100.0 * speed_norm ** 2)
    landing = 200.0 * landing_prox * landing_low_speed * both_legs

    # 5. 极轻的姿态正则化（避免持续自旋）
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.0001

    total_reward = (
        w_progress * progress
        + w_approach * approach_vel
        - w_speed_near * near_gate * speed_penalty
        + landing
        - w_ori * angle_penalty
    )

    components = {
        'progress': w_progress * progress,
        'approach_vel': w_approach * approach_vel,
        'near_speed_penalty': -w_speed_near * near_gate * speed_penalty,
        'landing': landing,
        'orientation': -w_ori * angle_penalty,
    }

    return float(total_reward), components
```