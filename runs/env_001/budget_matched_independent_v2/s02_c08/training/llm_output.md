# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对着陆台高度）
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
    distance_reward = -distance * 0.5  # 线性惩罚，每远离1单位扣0.5

    # 2. 速度惩罚：鼓励减速，尤其是接近目标时
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离动态调整速度惩罚权重：越近惩罚越大
    speed_weight = 0.3 + 0.7 * (1.0 / (1.0 + distance * 2.0))
    speed_penalty = -speed * speed_weight * 0.8

    # 3. 角度奖励：鼓励直立（角度为0表示直立）
    angle_penalty = -(angle ** 2) * 0.3  # 二次惩罚，偏离直立越远惩罚越大

    # 4. 角速度惩罚：鼓励稳定姿态
    ang_vel_penalty = -(ang_vel ** 2) * 0.2

    # 5. 接触奖励：鼓励双脚同时接触着陆台
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = both_contact * 2.0

    # 6. 动作惩罚：鼓励少用引擎（action 0=无引擎，1=左姿态，2=主引擎，3=右姿态）
    # 主引擎（action=2）惩罚最大，姿态引擎（1和3）中等，无引擎无惩罚
    if action == 0:
        action_penalty = 0.0
    elif action == 1 or action == 3:
        action_penalty = -0.3
    else:  # action == 2 (main engine)
        action_penalty = -0.6

    # 7. 速度变化惩罚：鼓励平滑运动，避免剧烈抖动
    # 用相邻两步位置差的变化来近似加速度
    acc_x = vx - obs[2]
    acc_y = vy - obs[3]
    jerk_penalty = -(acc_x ** 2 + acc_y ** 2) * 0.1

    # 8. 进度奖励：当非常接近目标且稳定时给予额外奖励
    settled_bonus = 0.0
    if distance < 0.3 and speed < 0.3 and abs(angle) < 0.1 and abs(ang_vel) < 0.1:
        settled_bonus = 5.0

    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    jerk_penalty + settled_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "jerk_penalty": jerk_penalty,
        "settled_bonus": settled_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器着陆任务，智能体需要从初始位置（靠近视口顶部中央）飞向并稳定降落在中央目标着陆台上。核心目标包括：快速接近目标、减速、保持直立姿态、平稳着陆、最小化引擎使用。

**信号选择与理由**：
- **位置信号（x, y）**：直接反映与目标的距离，是任务最核心的指标
- **速度信号（vx, vy）**：控制着陆速度，避免硬着陆
- **角度信号（angle, ang_vel）**：保持直立姿态是安全着陆的前提
- **接触信号（left_contact, right_contact）**：判断是否成功着陆在平台上
- **动作信号（action）**：惩罚引擎使用，鼓励节能

**奖励项设计理由**：
1. **距离奖励**：线性惩罚形式，简单有效，梯度稳定
2. **速度惩罚**：动态权重设计，越接近目标对速度越敏感，避免高速撞击
3. **角度惩罚**：二次形式，对偏离直立姿态给予非线性惩罚
4. **角速度惩罚**：抑制旋转，促进姿态稳定
5. **接触奖励**：二值奖励，鼓励双脚同时接触着陆台
6. **动作惩罚**：区分引擎类型，主引擎惩罚最大，鼓励使用姿态引擎微调
7. **抖动惩罚**：抑制加速度突变，促进平滑轨迹
8. **稳定奖励**：在完全满足着陆条件时给予大额奖励，加速收敛