# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # 水平位置（相对于目标）
    y = next_obs[1]          # 垂直位置（相对于着陆点高度）
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机体角度
    ang_vel = next_obs[5]    # 角速度
    left_contact = next_obs[6]   # 左支撑接触标志
    right_contact = next_obs[7]  # 右支撑接触标志
    
    # 从 obs 提取上一时刻信号（用于计算变化）
    prev_x = obs[0]
    prev_y = obs[1]
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]
    
    # 1. 距离奖励：鼓励接近目标
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.02 + 0.08 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed
    
    # 3. 接近速度奖励：鼓励向目标移动
    # 计算位置变化方向（从 obs 到 next_obs）
    dx = x - prev_x
    dy = y - prev_y
    # 计算朝向目标的方向（负方向表示朝向目标）
    target_direction_x = -x / (distance + 1e-8)
    target_direction_y = -y / (distance + 1e-8)
    # 速度在朝向目标方向上的投影
    approach_speed = vx * target_direction_x + vy * target_direction_y
    approach_reward = 0.05 * max(0, approach_speed)
    
    # 4. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2)
    
    # 5. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel ** 2)
    
    # 6. 接触奖励：鼓励安全着陆
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact
    
    # 7. 着陆状态综合奖励：当同时满足接触、低速度、小角度时给予额外奖励
    low_speed = 1.0 if speed < 0.5 else 0.0
    upright = 1.0 if abs(angle) < 0.2 else 0.0
    settled_reward = 1.0 * both_contact * low_speed * upright
    
    # 8. 动作惩罚：鼓励节能（根据动作类型）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎
        action_penalty = -0.02
    elif action in [1, 3]:  # 姿态引擎
        action_penalty = -0.01
    
    # 9. 进度奖励：随着训练进行，逐渐增加对精确着陆的重视
    progress_factor = training_progress  # 0.0 到 1.0
    precision_bonus = progress_factor * (0.5 * both_contact * low_speed * upright)
    
    # 计算总奖励
    total_reward = (distance_reward + speed_penalty + approach_reward + 
                    angle_penalty + ang_vel_penalty + contact_reward + 
                    settled_reward + action_penalty + precision_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'approach_reward': approach_reward,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'settled_reward': settled_reward,
        'action_penalty': action_penalty,
        'precision_bonus': precision_bonus,
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向并稳定着陆在中央目标区域。核心目标包括：快速接近目标、减速、保持稳定姿态、安全接触、节能。

**信号选择**：
- 使用 `x, y` 计算距离，这是最直接的接近度度量
- 使用 `vx, vy` 计算速度和接近速度，控制运动行为
- 使用 `angle, ang_vel` 控制姿态稳定性
- 使用 `left_contact, right_contact` 检测着陆状态
- 使用 `action` 施加节能惩罚

**奖励设计理由**：
1. **距离奖励**（负值）：持续引导智能体向目标移动，值域随距离线性变化，避免梯度消失
2. **速度惩罚**：根据距离动态调整权重，远处允许较快速度，近处强制减速，防止高速撞击
3. **接近速度奖励**：鼓励向目标方向移动的正向奖励，与速度惩罚形成平衡
4. **姿态惩罚**：平方形式惩罚大角度偏移，鼓励直立姿态
5. **角速度惩罚**：抑制旋转，促进稳定
6. **接触奖励**：二值奖励，鼓励双脚同时接触
7. **着陆综合奖励**：同时满足接触、低速、直立三个条件时给予高额奖励，明确最终目标
8. **动作惩罚**：区分主引擎和姿态引擎的能耗，鼓励节能
9. **进度奖励**：随训练进程逐渐增加对精确着陆的重视，实现课程学习效果