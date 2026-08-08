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

    # 从 obs 提取当前步信号（用于动作惩罚）
    x_curr = obs[0]
    y_curr = obs[1]
    vx_curr = obs[2]
    vy_curr = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.02 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1=左转, 2=主引擎, 3=右转）
    # 动作0是无引擎，不惩罚
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态调整引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎（推力大，惩罚更大）
        action_penalty = -0.05

    # 7. 进度奖励：当接近目标且速度低时给予额外奖励
    # 距离<0.5且速度<0.3视为"settled"
    settled_bonus = 0.0
    if distance < 0.5 and speed < 0.3 and both_contact > 0.5:
        settled_bonus = 1.0

    # 汇总
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + settled_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "settled_bonus": settled_bonus
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向中央目标点，并稳定着陆。核心目标包括：接近目标、减速、保持直立姿态、双脚同时接触着陆、最小化引擎使用。

**信号选择**：
- 使用`x, y`计算距离（连续梯度信号，每步都有意义）
- 使用`vx, vy`计算速度（减速是着陆关键）
- 使用`angle`和`ang_vel`控制姿态稳定性
- 使用`left_contact, right_contact`检测着陆状态
- 使用`action`惩罚引擎使用（任务要求最小化推力）

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：提供持续向目标靠近的梯度，避免稀疏奖励问题
2. **速度惩罚**（动态权重）：越接近目标惩罚越重，鼓励减速；远处允许较快移动
3. **姿态惩罚**（二次形式）：角度偏差越大惩罚越重，鼓励直立
4. **角速度惩罚**（二次形式）：抑制旋转，保持稳定
5. **接触奖励**（二值但高权重）：鼓励双脚同时接触，这是稳定着陆的标志
6. **动作惩罚**（固定值）：主引擎惩罚更大（0.05 vs 0.02），鼓励优先使用姿态引擎
7. **settled奖励**（条件触发）：当同时满足近距离、低速、双脚接触时给予额外奖励，引导最终稳定状态

**数值稳定性**：所有奖励项都在合理范围内（-1到1之间），避免极端值导致训练不稳定。