# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志
    
    # 从 obs 提取上一时刻的位置（用于计算速度变化惩罚）
    prev_x = obs[0]
    prev_y = obs[1]
    
    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -distance * 0.5  # 线性惩罚，每步减少距离
    
    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_penalty_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 2.0))
    speed_penalty = -speed * speed_penalty_weight * 0.8
    
    # 3. 姿态奖励：鼓励保持直立（角度接近0）
    angle_penalty = -abs(angle) * 0.3
    
    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -abs(ang_vel) * 0.2
    
    # 5. 接触奖励：鼓励双脚同时接触着陆点
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = both_contact * 1.0
    
    # 6. 动作惩罚：鼓励少用引擎（动作1、2、3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    elif action == 2:  # 主引擎
        action_penalty = -0.15
    
    # 7. 速度变化惩罚：鼓励平滑运动（避免剧烈抖动）
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    speed_change = abs(speed - prev_speed)
    smoothness_penalty = -speed_change * 0.1
    
    # 8. 接近目标时的速度奖励：当非常接近目标时，鼓励完全静止
    if distance < 0.3:
        stillness_bonus = -speed * 2.0  # 强烈惩罚速度
    else:
        stillness_bonus = 0.0
    
    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    smoothness_penalty + stillness_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'smoothness_penalty': smoothness_penalty,
        'stillness_bonus': stillness_bonus,
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向中央目标点，同时保持稳定姿态并尽量减少燃料消耗。任务要求"快速到达并稳定在目标点"。

**信号选择**：
- 使用`x_position`和`y_position`计算距离，这是最直接的任务进度指标
- 使用`x_velocity`和`y_velocity`计算速度，用于鼓励减速（着陆需要）
- 使用`body_angle`和`angular_velocity`控制姿态稳定性
- 使用`left_support_contact`和`right_support_contact`检测是否成功着陆
- 使用`action`信息惩罚引擎使用，鼓励节能

**奖励设计理由**：
1. **距离奖励**（线性负奖励）：每步惩罚当前距离，引导智能体持续向目标移动
2. **速度惩罚**（动态加权）：距离越近惩罚越重，避免高速撞击目标；远处允许一定速度
3. **姿态惩罚**：保持直立姿态是稳定着陆的前提
4. **角速度惩罚**：抑制旋转，促进稳定
5. **接触奖励**：双脚同时接触是成功着陆的标志
6. **动作惩罚**：区分主引擎（高消耗）和姿态引擎（低消耗），鼓励节能
7. **平滑度惩罚**：抑制速度突变，促进平稳飞行
8. **静止奖励**：在目标附近强烈惩罚速度，确保最终静止

**数值稳定性**：所有奖励项都在合理范围内（-2到2之间），避免极端值导致训练不稳定。使用线性函数而非指数函数，保持梯度稳定。