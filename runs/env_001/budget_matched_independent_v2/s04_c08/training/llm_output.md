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

    # 从 obs 提取当前状态用于速度变化计算
    vx_prev = obs[2]
    vy_prev = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，距离近时严格惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-distance * 2.0)))

    # 3. 姿态奖励：鼓励保持直立（角度为0）
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 燃料效率惩罚：惩罚使用引擎（动作1=左转, 2=主引擎, 3=右转）
    engine_used = 1.0 if action in [1, 2, 3] else 0.0
    fuel_penalty = -0.02 * engine_used

    # 7. 速度变化奖励：鼓励在接近目标时减速
    speed_prev = (vx_prev ** 2 + vy_prev ** 2) ** 0.5
    speed_change = speed_prev - speed  # 正数表示减速
    deceleration_reward = 0.03 * speed_change * (1.0 / (1.0 + distance * 0.5))

    # 汇总奖励
    total_reward = distance_reward + speed_penalty + angle_penalty + ang_vel_penalty + contact_reward + fuel_penalty + deceleration_reward

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "deceleration_reward": deceleration_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆点，并稳定着陆。核心目标包括：快速到达目标、减速至接近零速度、保持直立姿态、使用最少引擎推力。

**信号选择与理由**：
- 使用 `x, y` 计算距离，直接反映接近目标的进度
- 使用 `vx, vy` 计算速度，控制着陆时的动能
- 使用 `angle` 和 `ang_vel` 控制姿态稳定性
- 使用 `left_contact, right_contact` 检测是否稳定着陆
- 使用 `action` 检测引擎使用情况以惩罚燃料消耗

**奖励项设计**：
1. **距离奖励**：线性惩罚距离，提供稳定的梯度引导智能体向目标移动
2. **速度惩罚**：使用 sigmoid 函数根据距离动态调整惩罚强度——远处允许较快速度，近处严格惩罚速度，实现"先快后慢"的策略
3. **姿态奖励**：二次惩罚角度偏差，鼓励保持直立
4. **角速度惩罚**：二次惩罚角速度，促进姿态稳定
5. **接触奖励**：二值奖励鼓励双脚同时接触，实现稳定着陆
6. **燃料惩罚**：对任何引擎使用施加小惩罚，鼓励高效飞行
7. **减速奖励**：根据距离加权，鼓励在接近目标时主动减速