# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]          # x_position: 水平坐标相对于目标
    y = next_obs[1]          # y_position: 垂直坐标相对于着陆点高度
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志
    
    # 从 obs 提取当前状态用于动作惩罚
    current_x = obs[0]
    current_y = obs[1]
    
    # 计算距离和速度
    distance = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    
    # 1. 接近目标奖励：鼓励向目标移动
    # 使用负指数形式，距离越近奖励越大
    approach_reward = 2.0 * (2.718281828 ** (-0.5 * distance))
    
    # 2. 速度控制奖励：鼓励在接近目标时减速
    # 当距离较远时允许较高速度，距离近时惩罚高速
    speed_penalty = -0.5 * speed * (1.0 - 2.718281828 ** (-0.3 * distance))
    
    # 3. 姿态稳定奖励：鼓励保持直立姿态
    # 角度偏离垂直方向越少越好，角速度越小越好
    angle_penalty = -0.3 * (angle ** 2 + 0.1 * ang_vel ** 2)
    
    # 4. 着陆接触奖励：鼓励双脚同时接触
    both_contact = 1.0 if left_contact > 0.5 and right_contact > 0.5 else 0.0
    contact_reward = 1.0 * both_contact
    
    # 5. 动作效率惩罚：鼓励少用引擎
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.2
    elif action in [1, 3]:  # 姿态引擎消耗中等
        action_penalty = -0.1
    
    # 6. 进度奖励：根据训练进度调整探索-利用平衡
    # 早期更注重探索（接近目标），后期更注重精确控制
    exploration_bonus = 0.5 * (1.0 - training_progress) * (2.718281828 ** (-0.2 * distance))
    precision_bonus = 0.5 * training_progress * (1.0 - speed / (1.0 + speed))
    
    # 计算总奖励
    total_reward = (approach_reward + speed_penalty + angle_penalty + 
                    contact_reward + action_penalty + 
                    exploration_bonus + precision_bonus)
    
    components = {
        'approach_reward': approach_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'exploration_bonus': exploration_bonus,
        'precision_bonus': precision_bonus
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置移动到目标着陆点，同时控制速度、姿态和引擎使用。核心目标包括：快速接近目标、平稳减速、保持稳定姿态、安全着陆、最小化引擎消耗。

**信号选择**：
- 使用`x_position`和`y_position`计算距离，这是最直接的进度指标
- 使用`x_velocity`和`y_velocity`计算速度，用于控制减速
- 使用`body_angle`和`angular_velocity`评估姿态稳定性
- 使用接触标志检测是否成功着陆
- 使用`action`信息惩罚不必要的引擎使用

**奖励设计理由**：
1. **接近奖励**：使用负指数形式，在远距离时提供稳定梯度，近距离时奖励更密集，引导智能体向目标移动
2. **速度惩罚**：结合距离的衰减因子，允许远距离高速移动，但要求近距离减速，避免高速撞击
3. **姿态惩罚**：二次形式惩罚角度和角速度偏离，鼓励保持稳定直立姿态
4. **接触奖励**：二值奖励鼓励双脚同时接触，这是成功着陆的关键标志
5. **动作惩罚**：区分主引擎和姿态引擎的消耗，鼓励高效使用引擎
6. **进度调整**：利用`training_progress`参数，早期鼓励探索（接近目标），后期鼓励精确控制（减速和稳定）