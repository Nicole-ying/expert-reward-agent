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

    # 从 obs 提取上一时刻信号用于速度变化
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，接近时惩罚速度
    speed_penalty = -0.05 * speed * (1.0 / (1.0 + distance * 0.5))

    # 3. 角度奖励：鼓励保持直立（角度为0）
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时着地
    both_contact = left_contact * right_contact  # 1.0 if both, 0.0 otherwise
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=no_engine, 1=left, 2=main, 3=right
    engine_used = 1.0 if action != 0 else 0.0
    action_penalty = -0.02 * engine_used

    # 7. 速度变化奖励：鼓励平稳减速（负加速度）
    accel = ((vx - prev_vx) ** 2 + (vy - prev_vy) ** 2) ** 0.5
    smoothness_penalty = -0.01 * accel

    # 8. 进度奖励：当接近目标且速度低时给予额外奖励
    settled_bonus = 0.0
    if distance < 0.5 and speed < 0.5 and both_contact > 0.5:
        settled_bonus = 1.0

    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    smoothness_penalty + settled_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "smoothness_penalty": smoothness_penalty,
        "settled_bonus": settled_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器/着陆器控制任务，智能体需要从初始位置（靠近视口顶部中央）移动到中央目标平台，并稳定着陆。核心目标包括：快速接近目标、减速、保持稳定姿态、双脚同时接触平台、最小化引擎使用。

**信号选择**：
- 使用`x, y`（位置）计算距离，这是最直接的接近度度量
- 使用`vx, vy`（速度）控制着陆速度，避免硬着陆
- 使用`angle, ang_vel`（姿态）保持稳定，防止翻转
- 使用`left_contact, right_contact`（接触标志）检测是否成功着陆
- 使用`action`（动作）惩罚不必要的引擎使用

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：简单直接地引导智能体向目标移动，避免使用平方项导致远距离时梯度太大
2. **速度惩罚**（距离自适应）：远距离时允许较高速度以快速接近，近距离时强烈惩罚速度以确保安全着陆
3. **角度/角速度惩罚**（二次形式）：对称惩罚任何角度偏差，鼓励保持直立姿态
4. **接触奖励**（二值乘积）：只有双脚同时接触才给予奖励，避免单脚着陆的局部最优
5. **动作惩罚**（线性）：鼓励最小化引擎使用，符合节能目标
6. **平滑度惩罚**：惩罚加速度突变，鼓励平稳飞行轨迹
7. **结算奖励**：当满足所有着陆条件（近距离、低速、双脚接触）时给予额外奖励，明确指示成功状态

**数值稳定性**：所有奖励项的量级控制在0.1-1.0之间，避免极端值导致训练不稳定。使用线性或二次形式而非指数形式，防止梯度爆炸。