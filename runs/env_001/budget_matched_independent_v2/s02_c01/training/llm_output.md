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
    left_contact = next_obs[6]   # left_support_contact: 左侧接触标志
    right_contact = next_obs[7]  # right_support_contact: 右侧接触标志

    # 从 obs 提取当前步信号（用于动作惩罚）
    current_x = obs[0]
    current_y = obs[1]
    current_vx = obs[2]
    current_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近越强调减速
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance * 0.5))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励节能（动作0为无引擎，惩罚其他动作）
    action_penalty = -0.02 if action != 0 else 0.0

    # 7. 着陆成功奖励：当接近目标、速度小、姿态正、双脚接触时给予额外奖励
    near_target = distance < 0.3
    low_speed = speed < 0.2
    good_angle = abs(angle) < 0.15
    stable_landing = near_target and low_speed and good_angle and both_contact
    landing_bonus = 2.0 if stable_landing else 0.0

    # 8. 进度奖励：如果比上一步更接近目标，给予正向激励
    prev_distance = (current_x ** 2 + current_y ** 2) ** 0.5
    progress_reward = 0.05 * (prev_distance - distance)  # 距离减少为正

    # 汇总
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    landing_bonus + progress_reward)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'landing_bonus': landing_bonus,
        'progress_reward': progress_reward,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置飞向中央目标点，以稳定姿态着陆，同时最小化燃料消耗和时间。

**信号选择理由**：
- 使用 `x_position` 和 `y_position` 计算距离，这是最直接的任务进度指标
- 使用 `x_velocity` 和 `y_velocity` 计算速度，着陆需要减速
- 使用 `body_angle` 和 `angular_velocity` 控制姿态稳定性
- 使用 `left_support_contact` 和 `right_support_contact` 检测着陆状态
- 使用 `action` 惩罚引擎使用，鼓励节能

**奖励设计理由**：
1. **距离奖励**：线性惩罚确保每步都有梯度，引导向目标移动
2. **速度惩罚**：动态权重使靠近目标时更强调减速，避免高速撞击
3. **姿态和角速度惩罚**：二次形式对小偏差容忍，大偏差严厉惩罚
4. **接触奖励**：二值但结合其他条件使用，鼓励稳定双脚着陆
5. **动作惩罚**：轻微惩罚非零动作，鼓励节能
6. **着陆奖励**：组合条件触发，给予明确的正向信号
7. **进度奖励**：基于距离变化，提供细粒度引导

**数值稳定性**：所有奖励项控制在合理范围（-2到3之间），避免极端值。