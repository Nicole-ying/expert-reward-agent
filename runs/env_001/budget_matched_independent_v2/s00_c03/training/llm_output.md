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

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    dist = (x**2 + y**2) ** 0.5
    dist_reward = -0.1 * dist  # 线性惩罚距离，每步约 -0.1 到 -1.0

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx**2 + vy**2) ** 0.5
    # 根据距离调整速度惩罚权重：越近越强调减速
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + dist))  # 远时0.05，近时0.2
    speed_penalty = -speed_weight * speed

    # 3. 接近奖励：如果这一步更接近目标，给予正向奖励
    dist_prev = (x_prev**2 + y_prev**2) ** 0.5
    approach_reward = 0.5 * (dist_prev - dist)  # 接近时为正，远离时为负

    # 4. 姿态奖励：鼓励直立（角度接近0），着陆时更严格
    angle_penalty = -0.02 * (angle**2)  # 二次惩罚角度偏差
    # 当有接触时，额外惩罚大角度
    if left_contact > 0.5 or right_contact > 0.5:
        angle_penalty += -0.05 * (angle**2)

    # 5. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel**2)

    # 6. 着陆奖励：当两个支撑都接触且速度很小时给予奖励
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    low_speed = 1.0 if speed < 0.5 else 0.0
    landing_bonus = 2.0 * both_contact * low_speed

    # 7. 动作惩罚：鼓励少用引擎（action 0=无引擎，1=左，2=主，3=右）
    # 主引擎(2)惩罚最大，姿态引擎(1,3)中等，无引擎(0)无惩罚
    if action == 0:
        action_penalty = 0.0
    elif action == 2:
        action_penalty = -0.1
    else:  # action 1 or 3 (orientation engines)
        action_penalty = -0.05

    # 8. 进度自适应：随着训练进行，逐渐增加对精确着陆的要求
    progress_factor = 0.5 + 0.5 * training_progress  # 从0.5到1.0
    landing_bonus = landing_bonus * progress_factor
    angle_penalty = angle_penalty * (1.0 + 0.5 * training_progress)

    total_reward = (
        dist_reward +
        speed_penalty +
        approach_reward +
        angle_penalty +
        ang_vel_penalty +
        landing_bonus +
        action_penalty
    )

    components = {
        "dist_reward": dist_reward,
        "speed_penalty": speed_penalty,
        "approach_reward": approach_reward,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus,
        "action_penalty": action_penalty,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆点，并稳定着陆。核心目标包括：快速接近目标、减速、保持直立姿态、安全接触、节省燃料。

**信号选择**：
- `x_position`, `y_position`：直接反映与目标的距离，是核心导航信号
- `x_velocity`, `y_velocity`：控制速度是着陆的关键，需要减速
- `body_angle`, `angular_velocity`：姿态控制对稳定着陆至关重要
- `left_support_contact`, `right_support_contact`：检测是否成功着陆
- `action`：用于惩罚引擎使用，鼓励节能

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：提供持续的接近目标梯度，简单有效
2. **速度惩罚**（自适应权重）：远距离时允许较快速度，近距离时强制减速，避免高速撞击
3. **接近奖励**（差分奖励）：直接奖励每一步的进步，提供即时反馈
4. **姿态惩罚**（二次形式）：对称惩罚任何角度偏差，着陆时加重惩罚
5. **角速度惩罚**（二次形式）：抑制旋转，促进稳定
6. **着陆奖励**（条件组合）：仅在双支撑接触且速度足够低时给予，避免虚假着陆
7. **动作惩罚**（分级惩罚）：主引擎惩罚最大，姿态引擎中等，无引擎无惩罚
8. **进度自适应**：随着训练进行逐渐增加着陆精度要求，实现课程学习效果