# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆台高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步的状态（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]
    angle_prev = obs[4]
    ang_vel_prev = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.5 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx**2 + vy**2) ** 0.5
    # 当距离较远时允许一定速度，距离近时严格惩罚速度
    speed_penalty = -0.3 * speed * (1.0 / (1.0 + distance * 0.5))

    # 3. 接近奖励：如果距离在减小则给予正向奖励
    distance_prev = (x_prev**2 + y_prev**2) ** 0.5
    distance_delta = distance_prev - distance  # 正值表示接近
    approach_reward = 2.0 * max(0.0, distance_delta)

    # 4. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.2 * abs(angle) - 0.1 * abs(ang_vel)

    # 5. 接触奖励：鼓励双脚同时接触着陆台
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 3.0 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3都消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = -0.1 if action != 0 else 0.0

    # 7. 稳定性奖励：当接近目标且接触良好时，奖励低角速度
    stability_bonus = 0.0
    if distance < 0.5 and both_contact:
        stability_bonus = 1.0 * (1.0 - min(1.0, abs(ang_vel) * 5.0))

    # 8. 进度自适应：随着训练进行，逐渐增加对精度的要求
    progress_factor = 0.5 + 0.5 * training_progress
    precision_bonus = 0.0
    if distance < 0.3 and both_contact:
        precision_bonus = 2.0 * progress_factor * (1.0 - distance / 0.3)

    total_reward = (
        distance_reward
        + speed_penalty
        + approach_reward
        + angle_penalty
        + contact_reward
        + action_penalty
        + stability_bonus
        + precision_bonus
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "approach_reward": approach_reward,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "stability_bonus": stability_bonus,
        "precision_bonus": precision_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解：** 这是一个2D飞行器着陆任务，智能体需要从起始位置（靠近视口顶部中央）飞向并稳定降落在中央目标着陆台上。核心要求是：快速到达、精确着陆、减少燃料消耗、保持稳定姿态。

**信号选择与理由：**
- 使用 `x_position` 和 `y_position` 计算距离，这是最直接的目标导向信号
- 使用 `x_velocity` 和 `y_velocity` 控制速度，避免高速撞击
- 使用 `body_angle` 和 `angular_velocity` 控制姿态稳定性
- 使用 `left_support_contact` 和 `right_support_contact` 检测是否成功着陆
- 使用 `action` 施加燃料消耗惩罚

**奖励项设计：**
1. **距离奖励（-0.5*distance）：** 线性惩罚距离，持续引导向目标移动
2. **速度惩罚（-0.3*speed/(1+0.5*distance)）：** 自适应速度惩罚，远处允许高速，近处严格限制
3. **接近奖励（2.0*max(0, distance_delta)）：** 正向激励每步的接近行为，提供密集梯度
4. **姿态惩罚（-0.2*|angle| - 0.1*|ang_vel|）：** 鼓励保持直立稳定
5. **接触奖励（3.0*both_contact）：** 强奖励双脚同时接触着陆台，明确成功标准
6. **动作惩罚（-0.1 if action!=0）：** 鼓励最小化引擎使用，节省燃料
7. **稳定性奖励（1.0*(1-min(1,|ang_vel|*5))）：** 在接近且接触良好时奖励低角速度
8. **精度奖励（2.0*progress*(1-distance/0.3)）：** 随训练进度逐渐提高对精度的要求，避免过早满足

**数值稳定性：** 所有奖励项都控制在合理范围内，使用线性或饱和函数避免极端值。距离使用 `**0.5` 计算平方根，避免数值问题。