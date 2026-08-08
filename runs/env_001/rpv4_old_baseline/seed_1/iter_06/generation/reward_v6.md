```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 1. 主学习信号：增强的 potential shaping ----
    w_dist = 10.0
    w_speed = 5.0
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping_reward = potential_next - potential_cur

    # ---- 2. 姿态惩罚（角度不要过大） ----
    w_angle = 0.5
    angle_penalty = -w_angle * (angle**2)

    # ---- 3. 角速度惩罚（避免快速旋转） ----
    w_angvel = 0.1
    ang_vel_penalty = -w_angvel * (ang_vel**2)

    # ---- 4. 燃料效率惩罚 ----
    if action == 2:           # 主发动机
        fuel_penalty = -0.15
    elif action in (1, 3):    # 左/右姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 5. 着陆成功大奖励 ----
    # 条件：双腿接触，速度、位置、角度都十分接近零
    contact_success = nleft and nright and (next_speed < 0.5) and (abs(nx) < 0.3) and (abs(ny) < 0.3) and (abs(nangle) < 0.1)
    success_bonus = 50.0 if contact_success else 0.0

    # ---- 6. 猛烈着陆惩罚（双腿接触但不安全） ----
    # 如果双腿接触但速度过高或角度过大，视为 crash
    crash_condition = nleft and nright and ((next_speed > 1.5) or (abs(nangle) > 0.3))
    crash_penalty = -10.0 if crash_condition else 0.0

    # ---- 7. 边界危险惩罚（防止飞出场外） ----
    # 横向边界：避免水平飞出视口
    boundary_x_penalty = 0.0
    if abs(nx) > 2.0:
        boundary_x_penalty += -10.0
    if abs(nx) > 4.0:
        boundary_x_penalty += -40.0  # 累计 -50

    # 纵向下边界：避免掉出视图下方（撞击地面以外区域）
    boundary_y_penalty = 0.0
    if ny < -1.0:
        boundary_y_penalty += -10.0
    if ny < -3.0:
        boundary_y_penalty += -40.0

    total_reward = (
        shaping_reward
        + angle_penalty
        + ang_vel_penalty
        + fuel_penalty
        + success_bonus
        + crash_penalty
        + boundary_x_penalty
        + boundary_y_penalty
    )

    components = {
        "shaping_reward": shaping_reward,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "fuel_penalty": fuel_penalty,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty,
        "boundary_x_penalty": boundary_x_penalty,
        "boundary_y_penalty": boundary_y_penalty,
    }

    return float(total_reward), components
```