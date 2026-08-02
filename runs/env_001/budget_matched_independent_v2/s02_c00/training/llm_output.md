# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步信号（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速，尤其在接近目标时
    speed = (vx**2 + vy**2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励直立姿态（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.02 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 接近速度奖励：鼓励向目标移动（速度方向指向目标）
    # 计算速度在指向目标方向上的投影
    if distance > 0.01:
        # 指向目标的单位向量
        target_dir_x = -x / distance
        target_dir_y = -y / distance
        # 速度在目标方向上的投影（正表示朝向目标）
        approach_speed = vx * target_dir_x + vy * target_dir_y
        # 奖励朝向目标的速度，惩罚远离目标的速度
        approach_reward = 0.02 * max(approach_speed, 0.0)
    else:
        approach_reward = 0.0

    # 7. 动作惩罚：鼓励少用引擎（动作1,2,3都消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    # 8. 进度奖励：如果距离显著减小，给予额外奖励
    prev_distance = (x_prev**2 + y_prev**2) ** 0.5
    distance_improvement = prev_distance - distance
    progress_reward = 0.1 * max(distance_improvement, 0.0)

    # 汇总奖励
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        contact_reward +
        approach_reward +
        action_penalty +
        progress_reward
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "approach_reward": approach_reward,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向并稳定降落在中央目标点。核心目标包括：快速接近目标、减速、保持直立姿态、双脚同时接触着陆、最小化燃料消耗。

**信号选择**：
- 使用`x_position`和`y_position`计算距离，这是最直接的接近度量
- 使用`x_velocity`和`y_velocity`计算速度，用于鼓励减速和朝向目标运动
- 使用`body_angle`和`angular_velocity`鼓励直立稳定姿态
- 使用`left_support_contact`和`right_support_contact`检测成功着陆
- 使用`action`计算燃料消耗惩罚

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：提供持续的接近梯度，避免稀疏奖励问题
2. **速度惩罚**（自适应权重）：在远处允许较高速度，近处强制减速，模拟真实着陆需求
3. **角度和角速度惩罚**（二次形式）：对偏离直立姿态给予平滑惩罚，鼓励稳定
4. **接触奖励**（二值）：明确奖励双脚同时接触的成功着陆状态
5. **接近速度奖励**（投影）：鼓励速度方向指向目标，避免无意义绕圈
6. **动作惩罚**（线性）：鼓励最小化引擎使用，实现燃料效率目标
7. **进度奖励**（正增量）：奖励每步的距离改善，提供即时反馈

**数值稳定性**：所有奖励项的量级控制在0.1-1.0范围内，避免极端值。使用`**0.5`代替`sqrt`，使用`max`确保非负值。