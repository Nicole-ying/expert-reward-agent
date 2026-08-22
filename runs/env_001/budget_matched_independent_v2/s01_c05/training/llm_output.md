# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-action state)
    x = next_obs[0]          # horizontal position relative to target
    y = next_obs[1]          # vertical position relative to pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body orientation angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]   # left support contact flag (0 or 1)
    right_contact = next_obs[7]  # right support contact flag (0 or 1)
    
    # Distance to target (target is at origin)
    distance = (x**2 + y**2) ** 0.5
    
    # Speed magnitude
    speed = (vx**2 + vy**2) ** 0.5
    
    # 1. Proximity reward: encourage being close to target
    # Use a smooth exponential to provide gradient at all distances
    proximity_reward = 2.718281828 ** (-2.0 * distance)
    
    # 2. Velocity penalty: penalize high speed, especially when close to target
    # Scale penalty by distance so it's stronger near target
    velocity_penalty = -0.5 * speed * (1.0 + 2.0 * (2.718281828 ** (-distance)))
    
    # 3. Orientation reward: encourage upright orientation (angle=0 means upright)
    # Penalize deviation from zero angle
    orientation_penalty = -0.3 * (angle ** 2)
    
    # 4. Angular velocity penalty: discourage spinning
    ang_vel_penalty = -0.2 * (ang_vel ** 2)
    
    # 5. Contact bonus: reward stable contact with both supports on the pad
    # Both contacts active indicates successful landing
    contact_bonus = 1.0 * left_contact * right_contact
    
    # 6. Action penalty: discourage unnecessary engine use
    # Action 0 = no engine, actions 1-3 use some thrust
    action_penalty = -0.1 if action != 0 else 0.0
    
    # 7. Landing completion bonus: large reward when settled on pad
    # Settled = close to target, low speed, upright, both contacts
    settled = (distance < 0.3) and (speed < 0.1) and (abs(angle) < 0.1) and (left_contact > 0.5) and (right_contact > 0.5)
    completion_bonus = 5.0 if settled else 0.0
    
    # Sum all components
    total_reward = (proximity_reward + velocity_penalty + orientation_penalty + 
                    ang_vel_penalty + contact_bonus + action_penalty + completion_bonus)
    
    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "completion_bonus": completion_bonus
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）移动到目标着陆点（中央的着陆台），同时控制姿态、降低速度，最终稳定着陆。动作空间包含4个离散动作：无引擎、左姿态引擎、主引擎、右姿态引擎。

**信号选择**：
- 使用`x, y`计算到目标的距离，这是核心导航信号
- 使用`vx, vy`计算速度，着陆需要减速
- 使用`angle`和`ang_vel`控制姿态稳定性
- 使用`left_contact, right_contact`检测是否成功着陆

**奖励设计理由**：
1. **proximity_reward**：指数衰减形式，距离越近奖励越大，提供连续梯度引导智能体向目标移动
2. **velocity_penalty**：惩罚高速，且惩罚强度随距离减小而增大（通过`1+2*exp(-distance)`因子），鼓励接近目标时减速
3. **orientation_penalty**：二次惩罚角度偏差，鼓励保持直立姿态
4. **ang_vel_penalty**：二次惩罚角速度，抑制旋转
5. **contact_bonus**：两个支撑点都接触时给予奖励，鼓励稳定着陆
6. **action_penalty**：对使用引擎的动作施加小惩罚，鼓励节能
7. **completion_bonus**：当满足所有着陆条件（近距离、低速、直立、双接触）时给予大额奖励，明确任务完成信号