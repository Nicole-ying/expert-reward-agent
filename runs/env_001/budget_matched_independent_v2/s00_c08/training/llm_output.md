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

    # 从 obs 提取上一时刻的速度用于加速度惩罚
    prev_vx = obs[2]
    prev_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.5 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速靠近目标，同时避免过大速度
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：远时允许较快，近时要求慢速
    speed_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 0.5))  # 近处权重高
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 1.0 * both_contact

    # 6. 动作效率惩罚：鼓励少用引擎（action 1,2,3 都消耗燃料）
    # action 0 是无引擎，其他动作都有燃料消耗
    fuel_penalty = -0.05 if action != 0 else 0.0

    # 7. 加速度惩罚：鼓励平滑运动（避免剧烈抖动）
    accel = ((vx - prev_vx) ** 2 + (vy - prev_vy) ** 2) ** 0.5
    accel_penalty = -0.1 * accel

    # 8. 着陆奖励：当接近目标且速度很小时给予额外奖励
    near_target = distance < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(angle) < 0.1
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + ang_vel_penalty +
                    contact_reward + fuel_penalty + accel_penalty + landing_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty,
        'accel_penalty': accel_penalty,
        'landing_bonus': landing_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置飞到中央目标点并稳定着陆。核心目标包括：1) 快速接近目标 2) 减速并稳定 3) 保持直立姿态 4) 双脚同时接触着陆 5) 节省燃料。

**信号选择理由**：
- `x_position`, `y_position`：直接反映与目标的距离，是核心导航信号
- `x_velocity`, `y_velocity`：控制速度是安全着陆的关键
- `body_angle`, `angular_velocity`：姿态稳定是成功着陆的前提
- `left_support_contact`, `right_support_contact`：判断是否成功着陆
- `action`：用于燃料惩罚，鼓励高效使用引擎

**奖励项设计理由**：
1. **距离奖励**：线性惩罚，提供稳定的梯度引导智能体向目标移动
2. **速度惩罚**：自适应权重，远处允许较快速度，近处强制减速，避免高速撞击
3. **姿态惩罚**：二次惩罚角度偏差，鼓励直立姿态
4. **角速度惩罚**：抑制旋转，保持稳定
5. **接触奖励**：鼓励双脚同时接触，这是成功着陆的标志
6. **燃料惩罚**：小惩罚鼓励高效使用引擎，避免无意义喷射
7. **加速度惩罚**：抑制剧烈抖动，鼓励平滑轨迹
8. **着陆奖励**：当所有条件满足时给予一次性奖励，强化成功着陆行为

**数值稳定性**：所有奖励项都在合理范围内，没有极端值或除零风险。使用线性或二次形式避免梯度爆炸。