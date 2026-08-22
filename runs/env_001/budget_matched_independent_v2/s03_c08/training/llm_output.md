# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-step state)
    x_pos = next_obs[0]       # horizontal position relative to target pad
    y_pos = next_obs[1]       # vertical position relative to pad height
    x_vel = next_obs[2]       # horizontal velocity
    y_vel = next_obs[3]       # vertical velocity
    angle = next_obs[4]       # body orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)
    
    # Also extract previous state for velocity change calculation
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    
    # ========== Reward Components ==========
    
    # 1. Distance reward: encourage approaching the target (0,0)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -0.1 * distance  # linear penalty on distance
    
    # 2. Velocity penalty: encourage low speed, especially near target
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Scale penalty by distance: closer to target = higher penalty for speed
    velocity_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))
    
    # 3. Orientation reward: encourage upright posture (angle near 0)
    angle_penalty = -0.02 * (angle ** 2)
    
    # 4. Angular velocity penalty: discourage spinning
    ang_vel_penalty = -0.01 * (ang_vel ** 2)
    
    # 5. Contact reward: reward stable contact with both supports
    both_contact = left_contact * right_contact  # 1 if both, 0 otherwise
    contact_reward = 0.5 * both_contact
    
    # 6. Fuel efficiency penalty: penalize engine usage
    # Action 2 is main engine, actions 1 and 3 are orientation engines
    engine_used = 1.0 if action in [1, 2, 3] else 0.0
    fuel_penalty = -0.02 * engine_used
    
    # 7. Progress bonus: reward reducing speed over time (smooth landing)
    # Compare current speed to previous speed
    prev_speed = (prev_x_vel ** 2 + prev_y_vel ** 2) ** 0.5
    speed_change = prev_speed - speed  # positive if slowing down
    progress_reward = 0.03 * max(0, speed_change)
    
    # 8. Settlement bonus: reward being stationary and upright near target
    settled = (distance < 0.3) and (speed < 0.1) and (abs(angle) < 0.1) and (abs(ang_vel) < 0.1)
    settlement_bonus = 1.0 if settled else 0.0
    
    # ========== Combine ==========
    total_reward = (distance_reward + velocity_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + fuel_penalty + 
                    progress_reward + settlement_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty,
        'progress_reward': progress_reward,
        'settlement_bonus': settlement_bonus,
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器/着陆器任务，智能体需要从初始位置（靠近视口顶部中央）移动到中央目标平台，并稳定着陆。目标是最小化到达时间、燃料消耗，同时保持稳定姿态。

**信号选择与理由**：
- **位置信号**（x_pos, y_pos）：直接衡量与目标的距离，是核心导航信号
- **速度信号**（x_vel, y_vel）：控制着陆速度，避免硬着陆
- **角度信号**（angle, ang_vel）：保持稳定姿态，防止翻转
- **接触信号**（left_contact, right_contact）：检测是否成功着陆在平台上
- **动作信号**（action）：用于惩罚引擎使用，鼓励燃料效率

**奖励项设计理由**：
1. **距离奖励**：线性惩罚，提供持续梯度引导智能体向目标移动
2. **速度惩罚**：根据距离动态调整权重——远处允许较高速度，近处严格要求低速，实现平滑减速
3. **角度惩罚**：二次形式惩罚偏离直立姿态，鼓励稳定飞行
4. **角速度惩罚**：抑制旋转，保持方向稳定
5. **接触奖励**：仅在两个支撑点都接触时给予奖励，鼓励完全着陆
6. **燃料惩罚**：任何引擎使用都施加小惩罚，鼓励节能
7. **进度奖励**：奖励减速行为，引导智能体学习平滑减速策略
8. **结算奖励**：当智能体完全稳定在目标附近时给予一次性奖励，明确任务完成信号