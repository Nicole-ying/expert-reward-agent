# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对着陆点高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步信号（用于动作惩罚）
    x_curr = obs[0]
    y_curr = obs[1]
    vx_curr = obs[2]
    vy_curr = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在原点 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其是接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.05 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.02 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚同时接触（稳定着陆）
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励少用引擎（动作1,2,3消耗燃料）
    # 动作0=无引擎，动作1=左姿态，动作2=主引擎，动作3=右姿态
    action_penalty = 0.0
    if action == 1 or action == 3:  # 姿态引擎
        action_penalty = -0.02
    elif action == 2:  # 主引擎
        action_penalty = -0.05

    # 7. 进度奖励：如果距离在减小，给予正向激励
    prev_distance = (x_curr ** 2 + y_curr ** 2) ** 0.5
    distance_delta = prev_distance - distance
    progress_reward = 0.2 * max(0.0, distance_delta)

    # 8. 着陆完成奖励：当接近目标且速度很小时给予额外奖励
    near_target = distance < 0.3
    low_speed = speed < 0.1
    stable_angle = abs(angle) < 0.1
    stable_ang_vel = abs(ang_vel) < 0.05
    settled = near_target and low_speed and stable_angle and stable_ang_vel and both_contact
    settle_bonus = 2.0 if settled else 0.0

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    progress_reward + settle_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'progress_reward': progress_reward,
        'settle_bonus': settle_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标平台，快速稳定着陆，同时最小化燃料消耗。

**信号选择理由**：
- `x_position` 和 `y_position`：直接反映与目标的距离，是核心导航信号
- `x_velocity` 和 `y_velocity`：控制速度是安全着陆的关键
- `body_angle` 和 `angular_velocity`：保持直立姿态是稳定着陆的前提
- `left_support_contact` 和 `right_support_contact`：检测是否成功着陆在平台上

**奖励项设计理由**：
1. **距离奖励**（线性惩罚）：提供持续梯度引导智能体向目标移动，避免稀疏奖励问题
2. **速度惩罚**（动态加权）：距离越近惩罚越重，鼓励智能体在接近目标时减速，避免高速撞击
3. **姿态惩罚**（二次形式）：对角度偏差进行二次惩罚，鼓励直立姿态，同时小偏差容忍度较高
4. **角速度惩罚**：抑制旋转，促进稳定
5. **接触奖励**：鼓励双脚同时接触平台，这是成功着陆的标志
6. **动作惩罚**：对使用引擎的动作施加小惩罚，鼓励节能策略
7. **进度奖励**：当距离减小时给予正向激励，强化正确的运动方向
8. **着陆完成奖励**：当所有条件满足时给予一次性大奖励，明确最终目标

**数值稳定性**：所有奖励项都控制在合理范围内（-2到2之间），避免极端值导致训练不稳定。使用线性或二次形式而非指数形式，保持梯度平滑。