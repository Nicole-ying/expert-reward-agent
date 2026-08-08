# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号（索引基于 observation_space.fields）
    x = next_obs[0]          # x_position: 水平坐标（相对于目标）
    y = next_obs[1]          # y_position: 垂直坐标（相对于着陆台高度）
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取当前步的信号（用于动作惩罚）
    current_x = obs[0]
    current_y = obs[1]
    current_vx = obs[2]
    current_vy = obs[3]

    # 1. 距离奖励：鼓励接近目标（目标在 (0,0)）
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚：鼓励减速（尤其在接近目标时）
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # 根据距离调整速度惩罚权重：越近惩罚越重
    speed_weight = 0.05 + 0.15 * (1.0 / (1.0 + distance))
    speed_penalty = -speed_weight * speed

    # 3. 姿态奖励：鼓励直立（角度接近0）
    angle_penalty = -0.02 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚：鼓励稳定
    ang_vel_penalty = -0.01 * (ang_vel ** 2)

    # 5. 接触奖励：鼓励双脚接触着陆台
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact

    # 6. 动作惩罚：鼓励节能（减少引擎使用）
    # action: 0=无引擎, 1=左姿态, 2=主引擎, 3=右姿态
    action_penalty = 0.0
    if action == 2:  # 主引擎消耗最大
        action_penalty = -0.1
    elif action in [1, 3]:  # 姿态引擎消耗中等
        action_penalty = -0.05
    # action == 0 无惩罚

    # 7. 进度奖励：如果双脚接触且速度很低，给予额外奖励
    settled = both_contact and speed < 0.1
    settle_bonus = 1.0 if settled else 0.0

    # 汇总奖励
    total_reward = (distance_reward + speed_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + action_penalty + 
                    settle_bonus)

    components = {
        'distance_reward': distance_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'action_penalty': action_penalty,
        'settle_bonus': settle_bonus
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器着陆任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆台，并稳定着陆。核心目标包括：快速接近目标、减速、保持直立姿态、双脚同时接触着陆台、最小化引擎使用。

**信号选择**：
- `x, y`（位置）：直接衡量与目标的距离，是核心优化信号
- `vx, vy`（速度）：控制速度是着陆的关键，尤其在接近目标时需要减速
- `angle, ang_vel`（姿态和角速度）：保持直立稳定是安全着陆的前提
- `left_contact, right_contact`（接触标志）：判断是否成功着陆

**奖励设计理由**：
1. **距离奖励**（线性惩罚）：提供持续的梯度引导智能体向目标移动，避免稀疏奖励问题
2. **速度惩罚**（自适应权重）：距离越近惩罚越重，鼓励智能体在接近目标时主动减速，避免高速撞击
3. **姿态惩罚**（二次形式）：对角度偏差进行平滑惩罚，鼓励直立姿态，同时允许小幅调整
4. **角速度惩罚**（二次形式）：抑制旋转，促进稳定
5. **接触奖励**（二值）：鼓励双脚同时接触，这是成功着陆的标志
6. **动作惩罚**（分级）：主引擎消耗最大，姿态引擎次之，无引擎无惩罚，鼓励节能策略
7. **结算奖励**（条件触发）：当双脚接触且速度极低时给予额外奖励，强化最终稳定状态

**数值平衡**：各奖励项的量级控制在相近范围（0.01-1.0），避免某一项主导训练过程。距离和速度是主要驱动信号，姿态和角速度是辅助稳定信号，接触和结算奖励是最终目标信号。