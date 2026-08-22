# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号 (索引基于 observation_space.fields)
    x = next_obs[0]          # x_position: 水平坐标相对于目标
    y = next_obs[1]          # y_position: 垂直坐标相对于着陆点高度
    vx = next_obs[2]         # x_velocity: 水平速度
    vy = next_obs[3]         # y_velocity: 垂直速度
    angle = next_obs[4]      # body_angle: 机体角度
    ang_vel = next_obs[5]    # angular_velocity: 角速度
    left_contact = next_obs[6]   # left_support_contact: 左支撑接触标志
    right_contact = next_obs[7]  # right_support_contact: 右支撑接触标志

    # 从 obs 提取上一时刻信号用于速度变化计算
    prev_vx = obs[2]
    prev_vy = obs[3]
    prev_angle = obs[4]
    prev_ang_vel = obs[5]

    # 1. 距离奖励: 鼓励接近目标 (目标在原点)
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # 线性惩罚距离

    # 2. 速度惩罚: 鼓励减速接近目标
    speed = (vx ** 2 + vy ** 2) ** 0.5
    speed_penalty = -0.05 * speed

    # 3. 姿态奖励: 鼓励保持直立 (角度为0)
    angle_penalty = -0.2 * (angle ** 2)  # 二次惩罚角度偏差

    # 4. 角速度惩罚: 鼓励稳定姿态
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 5. 着陆奖励: 当两个支撑都接触时给予正向奖励
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 2.0 * both_contact

    # 6. 动作效率惩罚: 鼓励少用引擎 (动作1,2,3消耗燃料)
    action_penalty = 0.0
    if action == 1 or action == 2 or action == 3:
        action_penalty = -0.02

    # 7. 速度变化惩罚: 鼓励平滑运动 (避免剧烈加速)
    acc_x = vx - prev_vx
    acc_y = vy - prev_vy
    jerk_penalty = -0.01 * ((acc_x ** 2) + (acc_y ** 2))

    # 8. 接近目标时速度衰减奖励: 鼓励在接近时减速
    # 当距离小且速度大时给予惩罚
    approach_quality = -0.1 * (distance * speed)  # 距离*速度越小越好

    # 9. 角度稳定性: 鼓励在接近目标时保持稳定角度
    angle_stability = -0.3 * (distance * (angle ** 2))  # 距离近时角度更重要

    # 10. 存活奖励: 鼓励持续探索
    survival_bonus = 0.01

    # 汇总奖励
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        contact_reward +
        action_penalty +
        jerk_penalty +
        approach_quality +
        angle_stability +
        survival_bonus
    )

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty,
        "jerk_penalty": jerk_penalty,
        "approach_quality": approach_quality,
        "angle_stability": angle_stability,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器轨迹优化任务，智能体需要从起始位置（靠近视口顶部中央）飞向中央目标着陆点，并稳定着陆。核心目标包括：快速接近目标、减速、保持稳定姿态、安全接触、节省燃料。

**信号选择**：
- 使用所有8个观测维度，因为它们都与任务目标直接相关
- 位置信号(x,y)用于计算距离奖励，引导智能体向目标移动
- 速度信号(vx,vy)用于惩罚高速接近，鼓励减速着陆
- 角度和角速度用于保持稳定姿态，这是安全着陆的前提
- 接触标志用于检测是否成功着陆在目标区域
- 动作信号用于惩罚不必要的引擎使用，鼓励节能

**奖励设计理由**：
1. **距离奖励**：线性惩罚距离，提供持续梯度引导智能体向目标移动
2. **速度惩罚**：线性惩罚速度，鼓励减速着陆
3. **姿态奖励**：二次惩罚角度偏差，因为角度偏差越大越危险
4. **角速度惩罚**：二次惩罚角速度，鼓励稳定姿态
5. **着陆奖励**：当两个支撑都接触时给予较大正向奖励，明确成功着陆信号
6. **动作惩罚**：轻微惩罚使用引擎的动作，鼓励节能
7. **速度变化惩罚**：惩罚加速度变化，鼓励平滑运动
8. **接近质量**：惩罚距离×速度的乘积，鼓励在接近目标时减速
9. **角度稳定性**：在距离近时更重视角度稳定性
10. **存活奖励**：小常数奖励，鼓励持续探索

**数值稳定性**：所有奖励项的量级控制在合理范围内（-10到+10之间），避免极端值导致训练不稳定。使用线性或二次形式而非指数形式，确保梯度稳定。