# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]       # x_position: 水平坐标（相对目标）
    y = next_obs[1]       # y_position: 垂直坐标（相对着陆点高度）
    vx = next_obs[2]      # x_velocity: 水平速度
    vy = next_obs[3]      # y_velocity: 垂直速度
    angle = next_obs[4]   # body_angle: 机体角度
    ang_vel = next_obs[5] # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取上一时刻的状态（用于计算速度变化等）
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -distance * 0.5  # 线性惩罚，每步减少距离

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 使用距离作为权重，越近越强调减速
    speed_penalty = -speed * (0.3 + 0.7 * (1.0 / (1.0 + distance * 2.0)))

    # 3. 角度奖励：鼓励保持直立（角度接近0）
    angle_penalty = -(angle ** 2) * 0.2  # 二次惩罚，对大幅倾斜更敏感

    # 4. 角速度惩罚：鼓励稳定，减少旋转
    ang_vel_penalty = -(ang_vel ** 2) * 0.1

    # 5. 着陆奖励：当两个支撑都接触时给予正向奖励
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    landing_bonus = both_contact * 2.0

    # 6. 稳定着陆额外奖励：接触时速度小且角度正
    stable_landing = both_contact * (1.0 if (speed < 0.5 and abs(angle) < 0.2) else 0.0)
    stable_landing_bonus = stable_landing * 3.0

    # 7. 燃料效率惩罚：惩罚使用引擎（动作1,2,3都消耗燃料）
    # 动作0是无引擎，其他动作都消耗燃料
    fuel_penalty = -0.1 if action != 0 else 0.0

    # 8. 速度变化惩罚：鼓励平滑运动，减少急加速/急减速
    accel = ((vx - prev_vx) ** 2 + (vy - prev_vy) ** 2) ** 0.5
    jerk_penalty = -accel * 0.05

    # 9. 角度变化惩罚：鼓励平滑旋转
    angle_change = abs(angle - prev_angle)
    angle_jerk_penalty = -angle_change * 0.1

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + landing_bonus + stable_landing_bonus +
                    fuel_penalty + jerk_penalty + angle_jerk_penalty)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'landing_bonus': landing_bonus,
        'stable_landing_bonus': stable_landing_bonus,
        'fuel_penalty': fuel_penalty,
        'jerk_penalty': jerk_penalty,
        'angle_jerk_penalty': angle_jerk_penalty,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向并稳定降落在中央目标点。核心目标包括：快速接近目标、减速、保持稳定姿态、安全着陆、节省燃料。

**信号选择与理由**：
- **位置信号 (x, y)**：直接反映与目标的距离，是任务的核心指标
- **速度信号 (vx, vy)**：控制着陆速度是关键，尤其在接近目标时需要减速
- **角度信号 (angle, ang_vel)**：保持稳定姿态是安全着陆的前提
- **接触信号 (left_contact, right_contact)**：判断是否成功着陆的唯一依据
- **动作信号**：用于惩罚燃料消耗，鼓励高效飞行

**奖励项设计**：
1. **距离奖励**：线性负奖励，持续引导智能体向目标移动
2. **速度惩罚**：结合距离权重，在远处允许较快速度，近处强制减速
3. **角度/角速度惩罚**：二次形式，对大幅倾斜给予更强惩罚
4. **着陆奖励**：二值奖励，鼓励完成着陆动作
5. **稳定着陆奖励**：在着陆基础上要求低速度和正角度，防止粗暴着陆
6. **燃料惩罚**：对非零动作施加小惩罚，鼓励节能
7. **平滑性惩罚**：惩罚速度和角度的突变，促进稳定飞行

**数学形式选择**：使用线性、二次和条件组合，平衡各目标的优先级。所有惩罚项的量级通过系数控制在合理范围内，避免某一项主导训练。