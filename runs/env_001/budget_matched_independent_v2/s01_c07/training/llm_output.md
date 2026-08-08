# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆台高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取上一时刻的位置（用于计算速度变化）
    prev_x = obs[0]
    prev_y = obs[1]

    # 1. 距离奖励：鼓励接近目标
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -distance * 0.5  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近越强调减速
    speed_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 0.5))
    speed_penalty = -speed * speed_weight * 0.3

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -(angle ** 2) * 0.2  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -(ang_vel ** 2) * 0.1

    # 5. 接触奖励：鼓励安全着陆（两个支撑点都接触）
    contact_bonus = 0.0
    if left_contact > 0.5 and right_contact > 0.5:
        contact_bonus = 1.0  # 两个支撑点都接触时给予奖励
    elif left_contact > 0.5 or right_contact > 0.5:
        contact_bonus = 0.3  # 单点接触给予部分奖励

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.2
    elif action == 1 or action == 3:  # 姿态引擎消耗中等
        action_penalty = -0.1

    # 7. 进度奖励：鼓励向目标移动
    # 计算位置变化方向是否朝向目标
    prev_distance = (prev_x ** 2 + prev_y ** 2) ** 0.5
    distance_delta = distance - prev_distance
    progress_reward = -distance_delta * 2.0  # 接近目标为正，远离为负

    # 8. 速度方向奖励：鼓励速度指向目标
    if distance > 0.01:
        # 计算从当前位置指向目标的方向向量（目标在原点）
        target_dir_x = -x / distance
        target_dir_y = -y / distance
        # 速度方向与目标方向的一致性
        speed_dir_dot = (vx * target_dir_x + vy * target_dir_y) / (speed + 0.001)
        direction_reward = speed_dir_dot * 0.2  # 朝向目标为正，背离为负
    else:
        direction_reward = 0.0

    # 汇总奖励
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        contact_bonus +
        action_penalty +
        progress_reward +
        direction_reward
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward,
        "direction_reward": direction_reward,
    }

    return float(total_reward), components
```

# 设计说明

## 任务目标理解
这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向并稳定着陆在中央目标区域。核心目标包括：快速接近目标、减速、保持稳定姿态、安全接触，同时最小化引擎使用。

## 信号选择与理由
- **位置信号 (x, y)**：直接衡量与目标的距离，是任务的核心指标
- **速度信号 (vx, vy)**：控制着陆速度，避免硬着陆
- **角度和角速度**：保持稳定姿态，防止翻转
- **接触标志**：确认是否成功着陆
- **动作**：惩罚不必要的引擎使用，鼓励节能

## 奖励项设计
1. **距离奖励**：线性惩罚，简单直接地引导向目标移动
2. **速度惩罚**：动态权重，越接近目标惩罚越重，鼓励减速
3. **姿态奖励**：二次惩罚角度偏差，鼓励直立
4. **角速度惩罚**：二次惩罚，鼓励稳定
5. **接触奖励**：分级奖励，鼓励安全着陆
6. **动作惩罚**：根据引擎类型分级惩罚
7. **进度奖励**：基于位置变化，鼓励持续向目标移动
8. **方向奖励**：鼓励速度方向指向目标，提高接近效率

## 数学形式选择
- 使用线性惩罚避免梯度消失，同时配合二次惩罚处理需要精确控制的量（角度、角速度）
- 动态权重使奖励在不同任务阶段自适应调整
- 分级奖励（接触）提供明确的阶段性目标信号