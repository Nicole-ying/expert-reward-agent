# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步的信号（用于动作惩罚）
    x_curr = obs[0]
    y_curr = obs[1]
    vx_curr = obs[2]
    vy_curr = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_reward = -0.5 * dist  # 线性惩罚距离，每步梯度稳定

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，接近时惩罚速度
    speed_penalty = -0.3 * speed * (1.0 / (1.0 + dist * 0.5))

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.2 * abs(angle)  # 角度偏离惩罚

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.1 * abs(ang_vel)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    elif action == 2:  # 主引擎
        action_penalty = -0.15

    # 7. 进度奖励：当接近目标且速度低时给予额外奖励
    # 距离<0.5且速度<0.3视为"接近稳定"
    near_target = dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(angle) < 0.2
    if near_target and low_speed and stable_angle:
        progress_bonus = 1.0
    else:
        progress_bonus = 0.0

    # 汇总奖励
    total_reward = (dist_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_bonus)

    components = {
        "dist_reward": dist_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "progress_bonus": progress_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆点，并稳定着陆。核心目标包括：快速接近目标、减速、保持稳定姿态、安全接触、最小化引擎使用。

**信号选择**：
- 使用`x, y`计算距离（连续梯度信号，每步都有意义）
- 使用`vx, vy`计算速度（控制接近速度）
- 使用`angle`和`ang_vel`控制姿态稳定性
- 使用`left_contact, right_contact`检测着陆状态
- 使用`action`惩罚引擎使用

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：提供稳定的接近梯度，避免平方项导致远距离时梯度爆炸
2. **速度惩罚**（距离自适应）：远距离允许较高速度，近距离强制减速，避免高速撞击
3. **姿态惩罚**：鼓励保持直立姿态，为着陆做准备
4. **角速度惩罚**：抑制旋转，保持稳定
5. **接触奖励**：只有双脚同时接触才给予高奖励，鼓励正确着陆姿势
6. **动作惩罚**：主引擎惩罚较重（0.15），姿态引擎较轻（0.05），鼓励节能
7. **进度奖励**：当同时满足近距离、低速、小角度时给予一次性奖励，加速收敛

**数值稳定性**：所有奖励项都在合理范围内（-2到3之间），避免极端值。使用线性函数而非指数函数，防止梯度消失或爆炸。