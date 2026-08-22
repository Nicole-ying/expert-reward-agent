# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # x_position: 水平坐标相对于目标
    y = next_obs[1]          # y_position: 垂直坐标相对于着陆台高度
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志
    
    # 从 obs 提取上一时刻的位置用于计算速度变化（可选）
    prev_x = obs[0]
    prev_y = obs[1]
    
    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -distance * 0.5  # 线性惩罚距离
    
    # 2. 速度奖励：鼓励低速着陆，但允许接近时减速
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 当距离较远时允许较高速度，接近时惩罚速度
    speed_penalty = -speed * 0.3 * (1.0 / (1.0 + distance * 2.0))
    
    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -abs(angle) * 0.2
    
    # 4. 角速度奖励：鼓励稳定（角速度小）
    ang_vel_penalty = -abs(ang_vel) * 0.1
    
    # 5. 接触奖励：鼓励双脚同时接触着陆台
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = both_contact * 2.0
    
    # 6. 动作效率奖励：惩罚不必要的引擎使用
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎
        action_penalty = -0.1
    elif action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.05
    
    # 7. 进度奖励：鼓励向目标移动
    # 计算位置变化（从 obs 到 next_obs）
    prev_distance = (prev_x ** 2 + prev_y ** 2) ** 0.5
    distance_change = prev_distance - distance
    progress_reward = max(0.0, distance_change) * 2.0  # 接近目标时给予正奖励
    
    # 8. 着陆奖励：当双脚接触且接近目标且速度很小时给予额外奖励
    landing_bonus = 0.0
    if both_contact > 0.5 and distance < 0.5 and speed < 0.3:
        landing_bonus = 5.0
    
    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_reward + landing_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'progress_reward': progress_reward,
        'landing_bonus': landing_bonus,
    }
    
    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆台，并稳定着陆。核心目标包括：快速接近目标、降低速度、保持稳定姿态、安全接触着陆台、最小化引擎使用。

**信号选择与理由**：
- 使用 `x_position` 和 `y_position`（索引0,1）计算距离，这是最直接的接近度度量
- 使用 `x_velocity` 和 `y_velocity`（索引2,3）计算速度，用于鼓励减速着陆
- 使用 `body_angle`（索引4）和 `angular_velocity`（索引5）控制姿态稳定性
- 使用 `left_support_contact` 和 `right_support_contact`（索引6,7）检测着陆接触
- 使用 `action` 惩罚不必要的引擎使用，鼓励节能

**奖励项设计**：
1. **距离奖励**：线性惩罚距离，提供持续梯度引导接近目标
2. **速度惩罚**：根据距离动态调整权重，远处允许高速，近处强制减速
3. **姿态惩罚**：惩罚偏离直立角度，鼓励稳定飞行
4. **角速度惩罚**：抑制旋转，保持稳定
5. **接触奖励**：双脚同时接触时给予正奖励，鼓励正确着陆姿态
6. **动作惩罚**：对主引擎和姿态引擎使用施加小惩罚，鼓励节能
7. **进度奖励**：基于距离减少量给予正奖励，强化向目标移动的行为
8. **着陆奖励**：在满足着陆条件（双脚接触、接近目标、低速）时给予一次性大奖励

**数学形式选择**：采用线性惩罚为主，避免指数函数可能导致的数值不稳定。进度奖励使用 `max(0, change)` 确保只奖励正向进展。着陆奖励使用条件判断，在满足特定条件时给予明确的正向激励。