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
    left_contact = next_obs[6]  # left support contact flag
    right_contact = next_obs[7] # right support contact flag
    
    # 从 obs 提取上一时刻信号用于计算变化
    prev_x = obs[0]
    prev_y = obs[1]
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]
    
    # 距离目标点的距离
    distance = (x**2 + y**2) ** 0.5
    
    # 速度大小
    speed = (vx**2 + vy**2) ** 0.5
    
    # 1. 接近目标奖励：鼓励向目标移动
    # 使用负距离作为奖励，距离越小越好
    distance_reward = -distance * 0.1
    
    # 2. 速度控制奖励：鼓励在接近目标时减速
    # 当距离较远时允许较高速度，距离近时要求低速
    target_speed = 0.5 * (1.0 - 2.718281828 ** (-distance * 0.5))
    speed_penalty = -abs(speed - target_speed) * 0.05
    
    # 3. 着陆奖励：鼓励两个支撑点同时接触
    contact_reward = 0.0
    if left_contact > 0.5 and right_contact > 0.5:
        # 两个支撑点都接触时给予奖励，且速度越小奖励越大
        contact_reward = 1.0 * (1.0 - min(speed / 2.0, 1.0))
    
    # 4. 姿态稳定奖励：鼓励保持直立姿态（角度接近0）
    angle_penalty = -abs(angle) * 0.02
    
    # 5. 角速度惩罚：鼓励减少旋转
    ang_vel_penalty = -abs(ang_vel) * 0.01
    
    # 6. 引擎使用惩罚：根据动作类型惩罚
    # action: 0=no_engine, 1=left_orientation, 2=main_engine, 3=right_orientation
    engine_penalty = 0.0
    if action == 2:  # main engine
        engine_penalty = -0.05
    elif action == 1 or action == 3:  # orientation engines
        engine_penalty = -0.02
    
    # 7. 进度奖励：当距离明显减小时给予额外奖励
    prev_distance = (prev_x**2 + prev_y**2) ** 0.5
    progress_reward = max(0.0, (prev_distance - distance) * 0.2)
    
    # 8. 存活奖励：保持不终止的奖励
    alive_bonus = 0.01
    
    total_reward = (distance_reward + speed_penalty + contact_reward + 
                    angle_penalty + ang_vel_penalty + engine_penalty + 
                    progress_reward + alive_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'contact_reward': contact_reward,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'engine_penalty': engine_penalty,
        'progress_reward': progress_reward,
        'alive_bonus': alive_bonus
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置快速到达目标点并稳定着陆，同时最小化引擎使用。核心目标包括：接近目标、减速、保持稳定姿态、安全接触。

**信号选择**：
- 使用`x, y`计算距离，这是最直接的接近度量
- 使用`vx, vy`计算速度，用于控制减速行为
- 使用`angle, ang_vel`控制姿态稳定性
- 使用`left_contact, right_contact`检测着陆状态
- 使用`action`惩罚引擎使用

**奖励设计理由**：
1. **距离奖励**：负距离作为连续信号，每步都提供梯度
2. **速度控制**：根据距离动态调整目标速度，远距离允许高速，近距离要求低速，避免奖励漏洞
3. **接触奖励**：仅在两个支撑点都接触时给予，且与速度负相关，鼓励安全着陆
4. **姿态奖励**：惩罚角度偏差和角速度，保持稳定飞行
5. **引擎惩罚**：区分主引擎和姿态引擎的能耗
6. **进度奖励**：直接奖励向目标靠近的行为
7. **存活奖励**：鼓励持续尝试，避免过早终止