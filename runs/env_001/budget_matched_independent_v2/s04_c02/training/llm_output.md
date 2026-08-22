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

    # 从 obs 提取当前步信号（用于计算变化量）
    x_prev = obs[0]
    y_prev = obs[1]
    vx_prev = obs[2]
    vy_prev = obs[3]
    angle_prev = obs[4]
    ang_vel_prev = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚，每步减少距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx**2 + vy**2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.02 + 0.08 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 角度奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.05 * abs(angle)

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.02 * abs(ang_vel)

    # 5. 接触奖励：鼓励双脚同时接触着陆点
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    engine_used = 1.0 if action != 0 else 0.0
    fuel_penalty = -0.02 * engine_used

    # 7. 进度奖励：如果距离显著减小，给予额外奖励
    distance_prev = (x_prev**2 + y_prev**2) ** 0.5
    distance_delta = distance_prev - distance
    progress_reward = 0.2 * max(0, distance_delta)

    # 8. 速度变化奖励：如果速度在合理范围内减小，给予奖励
    speed_prev = (vx_prev**2 + vy_prev**2) ** 0.5
    speed_delta = speed_prev - speed
    # 只在接近目标时奖励减速
    if distance < 2.0:
        deceleration_reward = 0.1 * max(0, speed_delta)
    else:
        deceleration_reward = 0.0

    # 汇总
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + fuel_penalty + 
                    progress_reward + deceleration_reward)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty,
        'progress_reward': progress_reward,
        'deceleration_reward': deceleration_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向中央目标点，同时保持稳定姿态、减少速度，并最终安全着陆。任务要求"尽可能快"且"使用尽可能少的引擎推力"。

**信号选择与理由**：
- **位置信号 (x, y)**：核心信号，用于计算到目标的距离，是任务完成度的直接度量
- **速度信号 (vx, vy)**：控制着陆速度的关键，避免硬着陆
- **角度信号 (angle, ang_vel)**：保持稳定姿态的必要条件，防止翻转
- **接触信号 (left_contact, right_contact)**：检测是否成功着陆，双脚接触表示稳定着陆
- **动作信号 (action)**：用于惩罚引擎使用，鼓励节能

**奖励项设计**：
1. **距离奖励**：线性负奖励，持续引导智能体向目标移动
2. **速度惩罚**：动态权重，越接近目标惩罚越大，鼓励减速着陆
3. **角度/角速度惩罚**：鼓励保持直立稳定姿态
4. **接触奖励**：双脚同时接触时给予正奖励，鼓励成功着陆
5. **燃料惩罚**：任何引擎使用都给予小惩罚，鼓励节能
6. **进度奖励**：每步距离减少给予额外奖励，加速学习
7. **减速奖励**：接近目标时减速给予额外奖励，促进平稳着陆

**数学形式选择**：采用线性函数为主，避免指数爆炸；动态权重使奖励随任务阶段自适应变化；所有奖励项量级相近（0.01-0.5），避免某一项主导训练。