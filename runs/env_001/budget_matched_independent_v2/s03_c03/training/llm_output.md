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
    current_x = obs[0]
    current_y = obs[1]
    current_vx = obs[2]
    current_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.5 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近越强调减速
    speed_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 0.5))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0），着陆时尤为重要
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定，减少旋转
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励稳定着陆（两个支撑点都接触）
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_bonus = 2.0 * both_contact

    # 6. 动作惩罚：鼓励节能，减少不必要的引擎使用
    # 动作 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.3
    elif action in [1, 3]:  # 姿态引擎消耗中等
        action_penalty = -0.15
    # 动作0无惩罚

    # 7. 速度方向奖励：鼓励朝向目标运动
    if distance > 0.01:
        # 计算速度在指向目标方向上的投影（负值表示朝向目标）
        direction_to_target = -x / distance, -y / distance
        vel_toward_target = vx * direction_to_target[0] + vy * direction_to_target[1]
        # 朝向目标运动为正奖励，远离为负
        direction_reward = 0.3 * vel_toward_target
    else:
        direction_reward = 0.0

    # 8. 着陆稳定性奖励：当接近地面且速度低时给予额外奖励
    # y接近0表示接近着陆点高度
    height_factor = 1.0 / (1.0 + abs(y) * 2.0)
    low_speed = 1.0 / (1.0 + speed * 3.0)
    stability_bonus = 0.5 * height_factor * low_speed

    # 汇总奖励
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        contact_bonus +
        action_penalty +
        direction_reward +
        stability_bonus
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "direction_reward": direction_reward,
        "stability_bonus": stability_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆点，并稳定着陆。核心目标包括：快速接近目标、减速、保持稳定姿态、节能使用引擎。

**信号选择**：
- 使用`x_position`和`y_position`计算距离，这是最直接的目标接近度量
- 使用`x_velocity`和`y_velocity`计算速度，用于鼓励减速
- 使用`body_angle`和`angular_velocity`控制姿态稳定性
- 使用`left_support_contact`和`right_support_contact`检测成功着陆
- 使用`action`信息惩罚不必要的引擎使用

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：提供持续的梯度引导智能体向目标移动，比二值奖励更有效
2. **速度惩罚**（自适应权重）：距离越近惩罚越重，鼓励在接近目标时减速，避免高速撞击
3. **姿态奖励**（二次惩罚）：角度偏差越大惩罚越重，鼓励保持直立姿态
4. **角速度惩罚**（二次惩罚）：抑制旋转，促进稳定
5. **接触奖励**（二值奖励）：两个支撑点都接触时给予较大奖励，明确着陆成功信号
6. **动作惩罚**（固定值）：主引擎消耗最大，姿态引擎次之，鼓励节能
7. **方向奖励**（速度投影）：鼓励速度方向指向目标，帮助智能体学习正确的运动方向
8. **稳定性奖励**（乘积形式）：结合高度和速度，在接近地面且速度低时给予额外奖励，促进平稳着陆