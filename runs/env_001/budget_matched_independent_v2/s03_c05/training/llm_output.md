# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]      # x_position relative to target pad
    y = next_obs[1]      # y_position relative to pad height
    vx = next_obs[2]     # x_velocity
    vy = next_obs[3]     # y_velocity
    angle = next_obs[4]  # body_angle
    ang_vel = next_obs[5] # angular_velocity
    left_contact = next_obs[6]   # left support contact flag
    right_contact = next_obs[7]  # right support contact flag

    # 距离奖励：鼓励接近目标（目标在原点）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离，每步梯度稳定

    # 速度惩罚：鼓励减速靠近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离远时允许一定速度，距离近时强烈惩罚速度
    speed_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))

    # 姿态奖励：鼓励直立（角度为0），减少角速度
    angle_penalty = -0.02 * (angle ** 2 + ang_vel ** 2)

    # 接触奖励：鼓励双脚同时接触目标平台
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 0.5 * both_contact

    # 动作惩罚：鼓励少用引擎（动作1,2,3都消耗燃料）
    action_penalty = -0.01 if action != 0 else 0.0

    # 存活奖励：鼓励持续探索直到成功
    alive_bonus = 0.01

    # 总奖励
    total_reward = distance_reward + speed_penalty + angle_penalty + contact_reward + action_penalty + alive_bonus

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "alive_bonus": alive_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）快速到达并稳定在中央目标平台上，同时最小化引擎推力消耗。核心要求是：接近目标、减速、保持稳定姿态、安全接触。

**信号选择**：
- `x_position` 和 `y_position`：直接反映与目标的距离，是核心优化信号
- `x_velocity` 和 `y_velocity`：控制速度是稳定着陆的关键
- `body_angle` 和 `angular_velocity`：姿态稳定性对安全接触至关重要
- `left_support_contact` 和 `right_support_contact`：接触标志指示是否成功着陆

**奖励项设计**：
1. **距离奖励**（线性惩罚）：每步提供稳定的梯度引导智能体向目标移动，避免使用平方项导致远距离时梯度爆炸
2. **速度惩罚**（自适应权重）：距离远时允许较高速度（权重接近0.05），距离近时惩罚加重（权重可达0.15），鼓励减速接近
3. **姿态惩罚**：二次形式惩罚角度和角速度偏差，鼓励保持直立稳定
4. **接触奖励**：二值奖励鼓励双脚同时接触平台，这是成功着陆的标志
5. **动作惩罚**：对任何非零动作施加小惩罚，鼓励节能
6. **存活奖励**：小常数奖励防止智能体过早放弃

**数值稳定性**：所有奖励项的量级控制在0.1-1.0范围内，避免极端值。使用线性距离而非平方距离防止远距离时梯度爆炸。