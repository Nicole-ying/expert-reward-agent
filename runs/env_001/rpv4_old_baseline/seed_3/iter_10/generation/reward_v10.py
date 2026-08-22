def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact

    # 上一状态
    x_prev, y_prev = obs[0], obs[1]

    # 距离与速度
    dist_prev = (x_prev**2 + y_prev**2)**0.5
    dist_next = (x**2 + y**2)**0.5
    speed_norm = (xv**2 + yv**2)**0.5

    # 1. 位置奖励：单步进展 + 全局距离惩罚
    progress = dist_prev - dist_next
    proximity_bonus = -dist_next * 0.05
    position_reward = progress * 5.0 + proximity_bonus

    # 2. 速度惩罚：速度绝对值惩罚，接近目标时加重
    approach_factor = max(0.0, 1.0 - dist_next / 2.0)
    speed_penalty = speed_norm * (0.5 + 2.0 * approach_factor)

    # 3. 高度成本：仅当未稳固接触时惩罚高度，鼓励下降
    height_cost = 0.0
    if both_legs < 1.0:
        height_cost = abs(y) * 0.02

    # 4. 姿态正则化
    orientation_cost = (angle**2 + 0.1 * ang_vel**2) * 0.01

    # 5. 接触奖励：随接近目标而放大
    dist_factor = 1.0 + 2.0 * max(0.0, 1.0 - dist_next / 1.5)
    contact_reward = (5.0 * left_contact + 5.0 * right_contact + 15.0 * both_legs) * dist_factor

    # 6. 垂直速度控制：贴近地面时避免重着陆
    vy_penalty = 0.0
    if y < 0.5:
        vy_penalty = (yv**2) * 2.0

    # 7. 着陆大奖：满足条件时一次性高额奖励
    landing_bonus = 0.0
    if both_legs > 0.5 and dist_next < 1.0 and speed_norm < 1.0:
        landing_bonus = 200.0 * (1.0 - dist_next) * (1.0 - speed_norm)

    # 8. 引擎惩罚：仅惩罚主引擎
    engine_penalty = 0.002 if action == 2 else 0.0

    # 汇总
    total_reward = (
        position_reward
        - speed_penalty
        - height_cost
        - orientation_cost
        + contact_reward
        - vy_penalty
        + landing_bonus
        - engine_penalty
    )

    components = {
        'position_reward': position_reward,
        'speed_penalty': -speed_penalty,
        'height_cost': -height_cost,
        'orientation_cost': -orientation_cost,
        'contact_reward': contact_reward,
        'vy_penalty': -vy_penalty,
        'landing_bonus': landing_bonus,
        'engine_penalty': -engine_penalty,
    }

    return float(total_reward), components