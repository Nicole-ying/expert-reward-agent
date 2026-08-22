# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对着陆台高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步信号（用于动作惩罚）
    current_x = obs[0]
    current_y = obs[1]
    current_vx = obs[2]
    current_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.5 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近越强调减速
    speed_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 0.5))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚偏离直立

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触着陆台
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact

    # 6. 动作惩罚：鼓励节能（减少不必要的引擎使用）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    elif action == 2:  # 主引擎
        action_penalty = -0.15

    # 7. 进度奖励：如果距离显著减小，给予正向激励
    prev_distance = (current_x ** 2 + current_y ** 2) ** 0.5
    distance_delta = prev_distance - distance
    progress_reward = 0.3 * max(0.0, distance_delta)

    # 8. 稳定着陆奖励：当接近目标且速度很小时给予额外奖励
    stable_landing_bonus = 0.0
    if distance < 0.5 and speed < 0.3 and both_contact:
        stable_landing_bonus = 3.0

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_reward + stable_landing_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'progress_reward': progress_reward,
        'stable_landing_bonus': stable_landing_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆台，尽可能快地稳定着陆，同时最小化引擎使用。

**信号选择与理由**：
- 使用`x, y`（位置）和`vx, vy`（速度）作为核心信号，直接反映接近目标和减速的需求
- 使用`angle`和`ang_vel`控制姿态稳定性，这是安全着陆的关键
- 使用`left_contact, right_contact`检测是否成功着陆在目标台上
- 所有信号都来自`observation_space.fields`，没有使用未声明的维度

**奖励项设计**：
1. **距离奖励**：线性惩罚距离，提供持续梯度引导智能体向目标移动
2. **速度惩罚**：动态加权，越接近目标惩罚越重，鼓励减速
3. **角度/角速度惩罚**：二次形式惩罚偏离直立，鼓励稳定姿态
4. **接触奖励**：二值奖励鼓励双脚同时接触着陆台
5. **动作惩罚**：区分主引擎（高能耗）和姿态引擎（低能耗），鼓励节能
6. **进度奖励**：正向激励距离的减小，加速学习过程
7. **稳定着陆奖励**：在接近目标且速度低时给予额外奖励，强化成功着陆行为

**数值稳定性**：所有奖励项都在合理范围内（-10到10之间），避免极端值导致训练不稳定。使用`**0.5`代替`sqrt`，使用线性或二次形式避免指数爆炸。