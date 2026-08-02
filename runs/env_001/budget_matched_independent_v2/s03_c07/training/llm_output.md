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
    angle_curr = obs[4]
    ang_vel_curr = obs[5]

    # 1. 距离奖励：鼓励接近目标（目标在 (0,0)）
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_reward = -0.1 * dist  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离远时允许一定速度，距离近时惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + 2.718281828 ** (-dist * 0.5)))

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.02
    elif action in [1, 3]:  # 姿态引擎消耗中等
        action_penalty = -0.01

    # 7. 进度奖励：当接近目标且速度低时给予额外奖励
    near_target = 1.0 if dist < 0.3 else 0.0
    low_speed = 1.0 if speed < 0.2 else 0.0
    stable_angle = 1.0 if abs(angle) < 0.1 else 0.0
    settled_bonus = 0.3 * near_target * low_speed * stable_angle

    # 8. 着陆完成奖励：双脚接触且稳定在目标附近
    landing_complete = 1.0 if both_contact and near_target and low_speed and stable_angle else 0.0
    landing_reward = 1.0 * landing_complete

    # 汇总
    total_reward = (dist_reward + speed_penalty + angle_penalty + ang_vel_penalty +
                    contact_reward + action_penalty + settled_bonus + landing_reward)

    components = {
        "dist_reward": dist_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "settled_bonus": settled_bonus,
        "landing_reward": landing_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向中央目标点，并稳定着陆。核心目标包括：接近目标、降低速度、保持直立姿态、双脚同时接触地面，同时最小化引擎使用。

**信号选择**：
- 使用 `x_position` 和 `y_position` 计算距离，这是最直接的接近度度量
- 使用 `x_velocity` 和 `y_velocity` 计算速度，着陆任务需要减速
- 使用 `body_angle` 和 `angular_velocity` 控制姿态稳定性
- 使用 `left_support_contact` 和 `right_support_contact` 检测着陆状态
- 使用 `action` 施加引擎使用惩罚

**奖励设计理由**：
1. **距离奖励**：线性惩罚距离，提供持续的接近梯度
2. **速度惩罚**：使用 sigmoid 函数根据距离动态调整速度惩罚权重——远处允许较快速度，近处严格惩罚速度，避免高速撞击
3. **姿态惩罚**：二次惩罚角度偏差，鼓励直立
4. **角速度惩罚**：二次惩罚角速度，鼓励稳定
5. **接触奖励**：双脚同时接触时给予奖励，鼓励对称着陆
6. **动作惩罚**：主引擎消耗最大，姿态引擎次之，鼓励节能
7. **进度奖励**：当同时满足近距、低速、稳定姿态时给予额外奖励，引导平稳接近
8. **着陆完成奖励**：完全满足着陆条件时给予一次性奖励，明确最终目标

**避免奖励漏洞**：
- 速度惩罚与距离耦合，防止 agent 在远处就过度减速
- 姿态和角速度惩罚防止 agent 通过旋转来"取巧"
- 动作惩罚防止无意义地频繁点火
- 接触奖励需要双脚同时接触，防止单脚触地就停止学习