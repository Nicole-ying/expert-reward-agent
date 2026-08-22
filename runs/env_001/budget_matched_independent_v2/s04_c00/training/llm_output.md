# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步的信号（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx**2 + vy**2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励直立姿态（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定，减少旋转
    ang_vel_penalty = -0.02 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1=左转, 2=主引擎, 3=右转）
    # 动作0是无引擎，不惩罚
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎（推力大，惩罚更大）
        action_penalty = -0.05

    # 7. 进度奖励：如果向目标移动则给予正向奖励
    # 计算距离变化（负值表示接近）
    prev_distance = (x_prev**2 + y_prev**2) ** 0.5
    distance_delta = prev_distance - distance  # 正数表示接近
    progress_reward = 0.2 * max(0, distance_delta)

    # 8. 速度变化奖励：鼓励减速（速度减小）
    prev_speed = (vx_prev**2 + vy_prev**2) ** 0.5
    speed_delta = prev_speed - speed  # 正数表示减速
    deceleration_reward = 0.1 * max(0, speed_delta)

    # 汇总
    total_reward = (
        distance_reward
        + speed_penalty
        + angle_penalty
        + ang_vel_penalty
        + contact_reward
        + action_penalty
        + progress_reward
        + deceleration_reward
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward,
        "deceleration_reward": deceleration_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置快速到达中心目标点并稳定着陆，同时最小化引擎使用。

**信号选择与理由**：
- **位置信号 (x, y)**：直接反映与目标的距离，是核心优化目标
- **速度信号 (vx, vy)**：控制着陆速度，避免硬着陆
- **角度信号 (angle, ang_vel)**：保持稳定姿态，为着陆做准备
- **接触信号 (left_contact, right_contact)**：检测是否成功着陆在支撑面上

**奖励项设计**：
1. **距离奖励**：线性惩罚距离，提供持续的接近梯度
2. **速度惩罚**：根据距离动态调整权重，远处允许较快速度，近处强制减速
3. **角度/角速度惩罚**：二次形式惩罚偏离直立，鼓励稳定姿态
4. **接触奖励**：二值奖励鼓励双脚同时接触，标志成功着陆
5. **动作惩罚**：区分主引擎和姿态引擎，鼓励节能
6. **进度奖励**：正向奖励向目标移动的行为，强化正确方向
7. **减速奖励**：鼓励主动减速，配合速度惩罚形成完整的速度控制策略

**数学形式选择**：使用线性/二次惩罚和正向奖励的组合，避免极端值，保持数值稳定。动态权重让智能体在不同阶段有不同优化重点。