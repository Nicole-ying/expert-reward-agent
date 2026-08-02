# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # x_position relative to target
    y = next_obs[1]          # y_position relative to pad height
    vx = next_obs[2]         # x_velocity
    vy = next_obs[3]         # y_velocity
    angle = next_obs[4]      # body_angle
    ang_vel = next_obs[5]    # angular_velocity
    left_contact = next_obs[6]   # left support contact flag
    right_contact = next_obs[7]  # right support contact flag

    # 从 obs 提取上一时刻信号（用于计算变化）
    prev_x = obs[0]
    prev_y = obs[1]
    prev_vx = obs[2]
    prev_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许一定速度，距离近时强烈惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-distance * 2.0)))

    # 3. 接近奖励：当距离减小时给予正向奖励
    prev_distance = (prev_x ** 2 + prev_y ** 2) ** 0.5
    distance_delta = prev_distance - distance
    approach_reward = 0.5 * max(0.0, distance_delta)  # 仅奖励正向接近

    # 4. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2 + ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触目标区域
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.3 * both_contact

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=no_engine, 1=left, 2=main, 3=right
    action_penalty = -0.02 if action != 0 else 0.0

    # 7. 稳定奖励：当接近目标且速度很小时给予额外奖励
    settled_bonus = 0.0
    if distance < 0.5 and speed < 0.1 and abs(angle) < 0.1 and abs(ang_vel) < 0.1:
        settled_bonus = 0.5

    total_reward = (distance_reward + speed_penalty + approach_reward +
                    angle_penalty + contact_reward + action_penalty + settled_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "approach_reward": approach_reward,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "settled_bonus": settled_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器/着陆器控制任务，智能体需要从初始位置（靠近视口顶部中央）移动到中央目标平台，并稳定着陆。核心要求是：快速到达目标、减少速度、保持稳定姿态、安全接触、节省燃料。

**信号选择**：
- 使用`x, y`计算距离，这是最直接的接近度量
- 使用`vx, vy`计算速度，用于鼓励减速
- 使用`angle, ang_vel`评估姿态稳定性
- 使用`left_contact, right_contact`检测是否成功着陆
- 使用`action`惩罚引擎使用，鼓励节能

**奖励设计理由**：
1. **距离奖励**：线性惩罚距离，提供持续梯度引导智能体向目标移动
2. **速度惩罚**：使用sigmoid函数根据距离动态调整惩罚强度——远距离时允许较快速度，近距离时强烈惩罚速度，避免高速撞击
3. **接近奖励**：仅奖励距离减小的时刻，鼓励持续向目标移动，避免原地徘徊
4. **姿态惩罚**：惩罚角度和角速度偏离零值，鼓励保持直立稳定
5. **接触奖励**：奖励双脚同时接触目标区域，这是成功着陆的关键信号
6. **动作惩罚**：轻微惩罚使用引擎，鼓励节能策略
7. **稳定奖励**：当智能体已到达目标区域且完全稳定时给予额外奖励，加速收敛

**数值稳定性**：所有奖励项的量级控制在合理范围内（-1到1之间），避免梯度爆炸或消失。使用`max(0.0, ...)`确保接近奖励非负，使用sigmoid函数平滑过渡速度惩罚强度。