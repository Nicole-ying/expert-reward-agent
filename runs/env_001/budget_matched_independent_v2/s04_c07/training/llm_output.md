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

    # 从 obs 提取当前步的信号（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近越强调减速
    speed_weight = 0.02 + 0.08 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 接近速度奖励：鼓励向目标移动
    # 计算位置变化方向（从 obs 到 next_obs）
    dx = x - x_prev
    dy = y - y_prev
    # 如果向目标靠近（距离减小），给予正向奖励
    distance_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    distance_change = distance_prev - distance
    approach_reward = 0.5 * max(0.0, distance_change)

    # 4. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)  # 二次惩罚角度偏差

    # 5. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.02 * (ang_vel ** 2)

    # 6. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 1.0 * both_contact

    # 7. 动作惩罚：鼓励少用引擎（动作1,2,3都消耗燃料）
    # 动作0是no_engine，无惩罚；其他动作有惩罚
    action_penalty = -0.02 if action != 0 else 0.0

    # 8. 存活奖励：鼓励不触发终止条件（但权重较低）
    survival_bonus = 0.01

    # 汇总
    total_reward = (distance_reward + speed_penalty + approach_reward +
                    angle_penalty + ang_vel_penalty + contact_reward +
                    action_penalty + survival_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'approach_reward': approach_reward,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'survival_bonus': survival_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置快速到达中心目标点并稳定着陆，同时最小化燃料消耗。核心目标包括：接近目标、减速、保持直立姿态、双脚同时接触着陆。

**信号选择**：
- 使用`x_position`和`y_position`计算距离，这是最直接的接近度量
- 使用`x_velocity`和`y_velocity`计算速度，用于鼓励减速
- 使用`body_angle`和`angular_velocity`控制姿态稳定性
- 使用`left_support_contact`和`right_support_contact`检测着陆状态
- 使用`action`值惩罚引擎使用，鼓励节能

**奖励设计理由**：
1. **距离奖励**：线性惩罚距离，提供持续的接近梯度，避免稀疏奖励问题
2. **速度惩罚**：根据距离动态调整权重，远处允许较快速度，近处强调减速，防止高速撞击
3. **接近速度奖励**：直接奖励每步向目标靠近的行为，提供即时正向反馈
4. **姿态惩罚**：二次惩罚角度偏差，鼓励直立姿态，为着陆做准备
5. **角速度惩罚**：抑制旋转，保持稳定
6. **接触奖励**：双脚同时接触时给予较大奖励，明确着陆成功信号
7. **动作惩罚**：轻微惩罚非零动作，鼓励节能策略
8. **存活奖励**：微小常数奖励，避免agent通过自杀来逃避惩罚

**数值稳定性**：所有奖励项的量级控制在合理范围内（-10到10之间），避免极端值导致训练不稳定。使用线性或二次形式而非指数形式，保持梯度平滑。