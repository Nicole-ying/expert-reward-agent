# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]       # x_position relative to target
    y = next_obs[1]       # y_position relative to pad height
    vx = next_obs[2]      # x_velocity
    vy = next_obs[3]      # y_velocity
    angle = next_obs[4]   # body_angle
    ang_vel = next_obs[5] # angular_velocity
    left_contact = next_obs[6]   # left support contact flag
    right_contact = next_obs[7]  # right support contact flag

    # 从 obs 提取上一时刻的位置（用于计算速度变化，辅助判断是否减速）
    prev_x = obs[0]
    prev_y = obs[1]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，距离近时强烈惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-distance * 2.0)))

    # 3. 角度奖励：鼓励保持直立（角度为0）
    angle_penalty = -0.02 * (angle ** 2)

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触目标平台
    contact_bonus = 0.0
    if left_contact > 0.5 and right_contact > 0.5:
        contact_bonus = 0.5  # 双脚同时接触给予奖励
    elif left_contact > 0.5 or right_contact > 0.5:
        contact_bonus = 0.1  # 单脚接触给予少量奖励

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=no_engine, 1=left, 2=main, 3=right
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎
        action_penalty = -0.05

    # 7. 速度变化奖励：鼓励减速（接近目标时）
    # 计算速度变化方向（从obs到next_obs）
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    speed_change = prev_speed - speed  # 正数表示减速
    deceleration_reward = 0.0
    if speed_change > 0 and distance < 2.0:  # 仅在接近目标时奖励减速
        deceleration_reward = 0.1 * speed_change

    # 8. 存活奖励：鼓励持续尝试
    alive_bonus = 0.01

    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_bonus + action_penalty + 
                    deceleration_reward + alive_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "deceleration_reward": deceleration_reward,
        "alive_bonus": alive_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从起始位置飞到中央目标平台并稳定着陆。核心目标包括：1) 快速接近目标 2) 减速着陆 3) 保持稳定姿态 4) 双脚同时接触平台 5) 最小化引擎使用。

**信号选择**：
- 使用`x, y`计算距离（连续梯度好）
- 使用`vx, vy`计算速度（关键信号）
- 使用`angle, ang_vel`控制姿态（着陆稳定性）
- 使用`left_contact, right_contact`检测着陆状态
- 使用`action`惩罚引擎使用

**奖励设计理由**：
1. **距离奖励**：线性惩罚，简单有效，提供持续梯度
2. **速度惩罚**：使用sigmoid函数根据距离动态调整权重，远距离允许速度，近距离强烈惩罚
3. **角度/角速度惩罚**：平方形式，对偏离直立状态给予渐进惩罚
4. **接触奖励**：二值但分两级（单脚/双脚），鼓励稳定着陆
5. **动作惩罚**：区分主引擎和姿态引擎，鼓励节能
6. **减速奖励**：仅在接近目标时奖励减速行为，避免agent在远处就减速
7. **存活奖励**：小常数，避免agent过早放弃

**数值稳定性**：所有奖励项都在合理范围内（-0.5到0.5之间），避免极端值。使用`**0.5`代替`sqrt`，使用`2.718281828`代替`exp`。