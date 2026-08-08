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
    distance_reward = -0.1 * distance  # 线性惩罚距离，每步梯度稳定

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离远时允许一定速度，距离近时强烈惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-distance * 0.5)))

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:
        action_penalty = -0.02  # 姿态引擎小惩罚
    elif action == 2:
        action_penalty = -0.05  # 主引擎较大惩罚

    # 7. 进度奖励：当接近目标且速度低时给予额外奖励
    # 距离<0.5且速度<0.3视为"接近稳定"
    near_target = distance < 0.5
    low_speed = speed < 0.3
    stable_approach = 1.0 if (near_target and low_speed) else 0.0
    progress_reward = 0.3 * stable_approach

    # 总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty +
                    ang_vel_penalty + contact_reward + action_penalty + progress_reward)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向中央目标点，以稳定姿态着陆，同时最小化燃料消耗。观测空间包含位置、速度、角度、角速度和接触标志。

**信号选择与理由**：
- **距离**（x, y）：核心目标信号，每步都有梯度，引导智能体向目标移动
- **速度**（vx, vy）：控制接近速度，避免高速撞击，同时允许远距离时快速移动
- **角度和角速度**：确保着陆姿态稳定，避免翻转
- **接触标志**：直接反映着陆成功的关键条件
- **动作**：直接惩罚引擎使用，鼓励节能

**数学形式设计**：
1. **距离奖励**：线性惩罚（-0.1*distance），简单稳定，避免指数形式可能导致的梯度消失
2. **速度惩罚**：使用sigmoid-like函数根据距离动态调整权重——远距离时速度惩罚较轻（允许快速接近），近距离时惩罚加重（需要减速）
3. **角度惩罚**：二次形式（-0.02*angle²），对小幅偏差容忍，大幅偏差强烈惩罚
4. **角速度惩罚**：二次形式，鼓励稳定
5. **接触奖励**：二值奖励（0.5），仅在双脚同时接触时触发，作为着陆成功的正向激励
6. **动作惩罚**：区分主引擎（-0.05）和姿态引擎（-0.02），鼓励优先使用姿态调整
7. **进度奖励**：当距离<0.5且速度<0.3时给予额外奖励，强化"接近稳定"状态

**避免奖励漏洞**：距离奖励和速度惩罚共同作用，防止智能体原地不动（距离惩罚驱动移动）或高速飞过目标（速度惩罚限制接近速度）。动作惩罚防止无意义地消耗燃料。