# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-action state)
    x_pos = next_obs[0]       # horizontal position relative to target
    y_pos = next_obs[1]       # vertical position relative to pad height
    x_vel = next_obs[2]       # horizontal velocity
    y_vel = next_obs[3]       # vertical velocity
    angle = next_obs[4]       # body orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)

    # Distance to target (Euclidean distance in position space)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5

    # Speed magnitude
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5

    # --- Reward components ---

    # 1. Proximity reward: encourage getting close to target
    # Exponential decay with distance, scaled to be in [0, 1]
    proximity_reward = 2.718281828 ** (-2.0 * distance)

    # 2. Velocity penalty: penalize high speed, especially when close to target
    # Use a distance-weighted penalty: when far, allow some speed; when close, penalize more
    velocity_penalty = -0.5 * speed * (1.0 + 2.0 * (2.718281828 ** (-1.0 * distance)))

    # 3. Orientation reward: encourage upright orientation (angle near 0)
    # Penalize deviation from vertical, using cosine-like smooth penalty
    orientation_penalty = -0.3 * (angle ** 2)

    # 4. Angular velocity penalty: discourage spinning
    angular_penalty = -0.2 * (ang_vel ** 2)

    # 5. Contact bonus: reward stable contact with both supports on the pad
    # Both contacts active indicates settled state
    contact_bonus = 0.5 * (left_contact * right_contact)

    # 6. Action penalty: penalize engine usage to encourage fuel efficiency
    # action 0 = no engine (no penalty), actions 1-3 = engine firing
    action_penalty = -0.1 if action != 0 else 0.0

    # 7. Landing completion bonus: large reward when settled on target with both contacts
    # Condition: very close to target, low speed, both contacts active
    is_settled = (distance < 0.1) and (speed < 0.1) and (left_contact > 0.5) and (right_contact > 0.5)
    completion_bonus = 10.0 if is_settled else 0.0

    # Sum all components
    total_reward = (proximity_reward + velocity_penalty + orientation_penalty +
                    angular_penalty + contact_bonus + action_penalty + completion_bonus)

    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angular_penalty": angular_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "completion_bonus": completion_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器/着陆器任务，智能体需要从初始位置（靠近视口顶部中央）快速到达中央目标平台并稳定着陆，同时最小化引擎使用。关键要求是：接近目标、减速、保持稳定姿态、安全接触。

**信号选择与理由**：
- **位置信号**（x_pos, y_pos）：直接衡量与目标的距离，是核心导航信号
- **速度信号**（x_vel, y_vel）：控制着陆速度，避免硬着陆
- **姿态信号**（angle, ang_vel）：保持稳定方向，防止翻转
- **接触信号**（left_contact, right_contact）：检测是否成功着陆在平台上

**奖励项设计**：
1. **proximity_reward**：使用指数衰减函数，距离越近奖励越高，提供连续梯度引导智能体向目标移动
2. **velocity_penalty**：距离加权惩罚，远处允许一定速度（鼓励快速接近），近处严厉惩罚高速（防止硬着陆）
3. **orientation_penalty**：平方惩罚偏离垂直的角度，鼓励直立姿态
4. **angular_penalty**：平方惩罚角速度，抑制旋转
5. **contact_bonus**：当两个支撑点都接触时给予奖励，鼓励稳定着陆
6. **action_penalty**：对任何引擎使用施加小惩罚，鼓励燃料效率
7. **completion_bonus**：当同时满足近距离、低速、双接触时给予大额完成奖励，明确任务成功标准

**数值稳定性**：所有奖励项都在合理范围内，避免极端值；使用指数函数确保平滑梯度；惩罚项使用平方形式保证可微性。