# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (state after taking action)
    x_pos = next_obs[0]       # horizontal position relative to target pad
    y_pos = next_obs[1]       # vertical position relative to pad height
    x_vel = next_obs[2]       # horizontal velocity
    y_vel = next_obs[3]       # vertical velocity
    angle = next_obs[4]       # body orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)
    
    # Distance to target (target is at origin)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    
    # Speed magnitude
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    
    # Whether both legs are on the pad (settled condition)
    both_contact = left_contact * right_contact  # 1 if both contact, 0 otherwise
    
    # 1. Distance reward: encourage approaching the target
    # Exponential decay based on distance, scaled to be in [0, 1]
    distance_reward = 2.718281828 ** (-2.0 * distance)
    
    # 2. Velocity penalty: encourage reducing speed when near target
    # Use sigmoid-like weighting to only penalize speed when close
    near_target = 2.718281828 ** (-3.0 * distance)
    velocity_penalty = -0.5 * speed * near_target
    
    # 3. Orientation reward: encourage upright orientation (angle=0)
    # Penalize deviation from vertical, scaled by angular velocity
    orientation_penalty = -0.3 * (angle ** 2 + 0.1 * ang_vel ** 2)
    
    # 4. Contact reward: reward both legs on the pad
    contact_reward = 2.0 * both_contact
    
    # 5. Fuel efficiency penalty: penalize engine usage
    # Action 2 is main engine, actions 1 and 3 are orientation engines
    engine_used = 1.0 if action in [1, 2, 3] else 0.0
    fuel_penalty = -0.1 * engine_used
    
    # 6. Stability bonus: reward being settled (both contact + low speed + near target)
    stability_bonus = 3.0 * both_contact * (2.718281828 ** (-5.0 * speed)) * (2.718281828 ** (-2.0 * distance))
    
    # Sum all components
    total_reward = distance_reward + velocity_penalty + orientation_penalty + contact_reward + fuel_penalty + stability_bonus
    
    components = {
        "distance_reward": distance_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "stability_bonus": stability_bonus
    }
    
    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器/着陆器任务，智能体需要从起始位置（靠近视口顶部中央）快速到达中心目标平台并稳定着陆，同时最小化燃料消耗。

**信号选择与理由**：
- `x_pos, y_pos`：位置信号，用于计算到目标的距离，是核心导航信号
- `x_vel, y_vel`：速度信号，用于控制着陆速度，避免硬着陆
- `angle, ang_vel`：姿态信号，保持稳定朝向对安全着陆至关重要
- `left_contact, right_contact`：接触信号，判断是否成功着陆在平台上

**奖励项设计**：
1. **距离奖励**（指数衰减）：使用`exp(-2*distance)`提供平滑的接近梯度，鼓励持续向目标移动
2. **速度惩罚**（加权）：使用`near_target`权重（`exp(-3*distance)`），仅在接近目标时惩罚速度，避免远离目标时错误地减速
3. **姿态惩罚**：惩罚角度偏差和角速度，鼓励保持直立稳定姿态
4. **接触奖励**：当双腿都接触平台时给予正向奖励，明确指示成功着陆
5. **燃料惩罚**：对任何引擎使用施加小惩罚，鼓励节能
6. **稳定奖励**：组合条件（双腿接触+低速+近距离）给予额外奖励，鼓励完全稳定着陆状态

**数学形式选择**：指数函数用于距离和速度的加权，提供平滑且数值稳定的梯度；惩罚项使用二次形式（平方）确保对称性和可微性。