```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包状态
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ---- 势能塑造：同时驱动位置、速度、角度和水平偏移趋近目标 ----
    w_dist = 8.0
    w_speed = 4.0
    w_angle = 1.0
    w_x = 3.0          # 预防水平出界

    phi_cur = w_dist * dist + w_speed * speed + w_angle * abs(angle) + w_x * abs(x)
    phi_next = w_dist * next_dist + w_speed * next_speed + w_angle * abs(nangle) + w_x * abs(nx)
    shaping = phi_cur - phi_next

    # ---- 低空垂直速度引导：期望安全下降速度 ----
    vertical_guide = 0.0
    if ny < 1.2:
        desired_vy = -0.7 * (max(ny, 0.01) ** 0.5)
        vertical_guide = -1.5 * abs(nvy - desired_vy)

    # ---- 低空水平速度惩罚 ----
    horizontal_speed_penalty = 0.0
    if ny < 0.6:
        horizontal_speed_penalty = -2.0 * abs(nvx)

    # ---- 姿态与角速度惩罚 ----
    angle_penalty = -0.5 * abs(nangle)
    angvel_penalty = -0.05 * abs(nang_vel)

    # ---- 燃料效率 ----
    if action == 2:          # 主引擎
        fuel_penalty = -0.15
    elif action in (1, 3):   # 姿态引擎
        fuel_penalty = -0.03
    else:
        fuel_penalty = 0.0

    # ---- 双腿接触持续奖励 ----
    contact_reward = 0.0
    if nleft and nright:
        speed_factor = max(0.0, 1.0 - next_speed)
        angle_factor = max(0.0, 1.0 - abs(nangle) / 0.3)
        contact_reward = 3.0 * speed_factor * angle_factor

    # ---- 成功软着陆一次奖励 ----
    success_bonus = 0.0
    if nleft and nright and (next_speed < 0.5) and (abs(nangle) < 0.2):
        success_bonus = 60.0

    # 汇总
    total_reward = (shaping +
                    vertical_guide +
                    horizontal_speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    fuel_penalty +
                    contact_reward +
                    success_bonus)

    components = {
        "shaping": shaping,
        "vertical_guide": vertical_guide,
        "horizontal_speed_penalty": horizontal_speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "fuel_penalty": fuel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }

    return float(total_reward), components
```