# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs
    x_pos = next_obs[0]       # horizontal coordinate relative to target pad
    y_pos = next_obs[1]       # vertical coordinate relative to pad height
    x_vel = next_obs[2]       # horizontal linear velocity
    y_vel = next_obs[3]       # vertical linear velocity
    body_angle = next_obs[4]  # orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)
    
    # Also extract from obs for delta calculations
    prev_x_pos = obs[0]
    prev_y_pos = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_body_angle = obs[4]
    prev_ang_vel = obs[5]
    
    # 1. Distance reward: encourage approaching the target pad
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -0.1 * distance  # negative penalty proportional to distance
    
    # 2. Velocity penalty: encourage reducing speed, especially near target
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Scale velocity penalty by distance - closer to target = more penalty for speed
    velocity_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))
    
    # 3. Angular stability: encourage upright orientation and low angular velocity
    # Body angle 0 means upright, penalize deviation
    angle_penalty = -0.02 * (body_angle ** 2)
    angular_vel_penalty = -0.01 * (ang_vel ** 2)
    
    # 4. Contact reward: encourage stable contact with both supports
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact
    
    # 5. Progress reward: reward moving closer to target compared to previous step
    prev_distance = (prev_x_pos ** 2 + prev_y_pos ** 2) ** 0.5
    distance_delta = prev_distance - distance  # positive if moving closer
    progress_reward = 0.2 * max(0.0, distance_delta)
    
    # 6. Fuel efficiency penalty: penalize engine usage
    # Action 0 = no engine, 1 = left orientation, 2 = main, 3 = right orientation
    fuel_penalty = 0.0
    if action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.02
    elif action == 2:  # main engine
        fuel_penalty = -0.05
    
    # 7. Settling bonus: when very close to target with low velocity and both contacts
    settled = (distance < 0.3 and speed < 0.1 and abs(body_angle) < 0.1 and 
               abs(ang_vel) < 0.05 and both_contact > 0.5)
    settling_bonus = 1.0 if settled else 0.0
    
    # Sum all components
    total_reward = (distance_reward + velocity_penalty + angle_penalty + 
                    angular_vel_penalty + contact_reward + progress_reward + 
                    fuel_penalty + settling_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'contact_reward': contact_reward,
        'progress_reward': progress_reward,
        'fuel_penalty': fuel_penalty,
        'settling_bonus': settling_bonus,
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）到达并稳定在中央目标平台上，同时最小化燃料消耗。任务要求快速到达、减速、保持稳定姿态并安全接触。

**信号选择**：
- 使用`x_pos`和`y_pos`计算距离，这是最直接的目标导向信号
- 使用`x_vel`和`y_vel`计算速度，用于惩罚高速（尤其是接近目标时）
- 使用`body_angle`和`angular_velocity`鼓励姿态稳定
- 使用`left_contact`和`right_contact`检测是否稳定接触平台
- 使用`action`信息惩罚引擎使用，鼓励燃料效率

**奖励设计理由**：
1. **距离奖励**：负距离惩罚提供持续梯度，引导智能体向目标移动
2. **速度惩罚**：根据距离动态缩放速度惩罚，在远处允许较高速度，近处强制减速
3. **角度稳定性**：二次惩罚形式对偏离直立姿态给予递增惩罚
4. **接触奖励**：当两个支撑点都接触时给予正奖励，鼓励稳定着陆
5. **进度奖励**：基于距离变化的正向奖励，鼓励持续向目标移动
6. **燃料惩罚**：对引擎使用施加小惩罚，鼓励高效轨迹
7. **稳定奖励**：当所有条件满足时给予一次性大奖励，明确最终目标状态

这种设计避免了仅奖励速度导致的原地打转问题，也避免了仅奖励存活导致的停滞问题，通过多目标平衡引导智能体学习完整的"接近-减速-稳定-着陆"行为序列。