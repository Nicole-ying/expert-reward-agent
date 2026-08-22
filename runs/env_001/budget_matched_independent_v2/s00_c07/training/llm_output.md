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

    # 1. Proximity reward: encourage approaching the target
    # Exponential decay so reward increases as distance decreases
    proximity_reward = 2.718281828 ** (-2.0 * distance)

    # 2. Velocity penalty: discourage high speed, especially when near target
    # Scale penalty by distance: when far, allow some speed; when close, penalize heavily
    velocity_penalty = -0.5 * speed * (1.0 + 2.0 * (2.718281828 ** (-3.0 * distance)))

    # 3. Orientation reward: encourage upright orientation (angle near 0)
    # Penalize deviation from vertical, using cosine-like smooth penalty
    orientation_penalty = -0.3 * (angle ** 2)

    # 4. Angular velocity penalty: discourage spinning
    angular_penalty = -0.2 * (ang_vel ** 2)

    # 5. Contact bonus: reward stable contact with both supports on the pad
    # Both contacts active indicates successful landing
    contact_bonus = 1.0 * (left_contact * right_contact)

    # 6. Action penalty: discourage unnecessary engine use
    # action 0 = no engine (no penalty), actions 1-3 = engine firing (penalty)
    action_penalty = -0.1 if action != 0 else 0.0

    # 7. Small survival bonus to encourage staying alive (avoid termination)
    survival_bonus = 0.05

    # Sum all components
    total_reward = (proximity_reward + velocity_penalty + orientation_penalty +
                    angular_penalty + contact_bonus + action_penalty + survival_bonus)

    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angular_penalty": angular_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器/着陆器任务，智能体需要从初始位置（靠近视口顶部中央）快速到达中央目标平台并稳定着陆，同时最小化燃料消耗。关键要求是：接近目标、降低速度、保持稳定姿态、安全接触。

**信号选择**：
- 使用`next_obs`中的所有8个观测信号，因为它们直接相关于任务目标
- 位置信号（x_pos, y_pos）用于计算距离，这是核心导航信号
- 速度信号（x_vel, y_vel）用于控制着陆速度
- 角度和角速度用于姿态控制
- 接触标志用于检测成功着陆

**奖励设计理由**：
1. **proximity_reward**：指数衰减形式，在远距离时提供温和梯度，近距离时奖励急剧增加，引导智能体向目标移动
2. **velocity_penalty**：结合距离的惩罚，允许远距离时高速移动（探索），但接近目标时必须减速，防止硬着陆
3. **orientation_penalty**：平方惩罚角度偏差，鼓励保持直立姿态，避免侧翻
4. **angular_penalty**：平方惩罚角速度，抑制旋转，保持稳定
5. **contact_bonus**：只有两个支撑点都接触时才给予奖励，鼓励完全着陆而非单点接触
6. **action_penalty**：对任何引擎使用施加小惩罚，鼓励节能策略
7. **survival_bonus**：小常数奖励，防止智能体通过快速自杀来避免负奖励

**数值稳定性**：所有奖励项都在合理范围内（-1到2之间），避免极端值。使用`**0.5`代替`sqrt`，使用`2.718281828**`代替`exp`。