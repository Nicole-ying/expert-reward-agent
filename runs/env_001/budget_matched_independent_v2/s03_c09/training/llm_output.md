# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # 水平位置（相对于目标）
    y = next_obs[1]          # 垂直位置（相对于着陆台高度）
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机体角度
    ang_vel = next_obs[5]    # 角速度
    left_contact = next_obs[6]   # 左侧接触标志
    right_contact = next_obs[7]  # 右侧接触标志
    
    # 从 obs 提取上一时刻的状态用于速度变化计算
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # 1. 距离奖励：鼓励接近目标
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重，越近惩罚越大
    speed_penalty_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_penalty_weight * speed
    
    # 3. 加速度惩罚：鼓励平稳运动，减少燃料消耗
    accel = ((vx - prev_vx) ** 2 + (vy - prev_vy) ** 2) ** 0.5
    accel_penalty = -0.02 * accel
    
    # 4. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.05 * abs(angle)
    
    # 5. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -0.03 * abs(ang_vel)
    
    # 6. 接触奖励：鼓励双脚接触着陆台
    contact_bonus = 0.0
    if left_contact > 0.5 and right_contact > 0.5:
        contact_bonus = 0.5  # 双脚接触给予较大奖励
    elif left_contact > 0.5 or right_contact > 0.5:
        contact_bonus = 0.1  # 单脚接触给予小奖励
    
    # 7. 动作惩罚：鼓励少用引擎（节省燃料）
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎
        action_penalty = -0.05
    
    # 8. 着陆完成奖励：当接近目标且稳定时给予额外奖励
    completion_bonus = 0.0
    if (distance < 0.5 and speed < 0.3 and abs(angle) < 0.1 
        and abs(ang_vel) < 0.1 and left_contact > 0.5 and right_contact > 0.5):
        completion_bonus = 2.0
    
    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + accel_penalty + 
                   angle_penalty + ang_vel_penalty + contact_bonus + 
                   action_penalty + completion_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'accel_penalty': accel_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_bonus': contact_bonus,
        'action_penalty': action_penalty,
        'completion_bonus': completion_bonus
    }
    
    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置飞向中央目标着陆台，同时控制速度、姿态和燃料消耗。核心目标是快速、平稳、省燃料地完成着陆。

**信号选择理由**：
- 使用`x`和`y`计算距离，这是最直接的接近目标度量
- 使用`vx`和`vy`计算速度，着陆任务需要减速
- 使用`angle`和`ang_vel`控制姿态稳定性
- 使用接触标志判断是否成功着陆
- 使用动作值惩罚燃料消耗

**奖励设计逻辑**：
1. **距离奖励**：线性负奖励，引导智能体向目标移动
2. **速度惩罚**：根据距离动态调整权重，远处允许较快速度，近处强制减速
3. **加速度惩罚**：鼓励平滑运动，避免剧烈推力变化
4. **姿态惩罚**：鼓励保持直立姿态
5. **角速度惩罚**：鼓励姿态稳定
6. **接触奖励**：鼓励双脚接触着陆台，这是成功着陆的标志
7. **动作惩罚**：鼓励节省燃料，不同引擎消耗不同
8. **完成奖励**：当所有条件满足时给予较大奖励，加速收敛

**数学形式选择**：采用线性惩罚和奖励的组合，避免指数函数可能导致的数值不稳定。权重经过调参确保各分量量级相近，不会出现某个分量主导训练的情况。