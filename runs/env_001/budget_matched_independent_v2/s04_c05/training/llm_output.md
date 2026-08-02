# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取信号
    x = next_obs[0]       # 水平位置（相对于目标）
    y = next_obs[1]       # 垂直位置（相对于着陆台高度）
    vx = next_obs[2]      # 水平速度
    vy = next_obs[3]      # 垂直速度
    angle = next_obs[4]   # 机体角度
    ang_vel = next_obs[5] # 角速度
    left_contact = next_obs[6]   # 左侧支撑接触标志
    right_contact = next_obs[7]  # 右侧支撑接触标志

    # 距离目标中心的距离
    distance = (x ** 2 + y ** 2) ** 0.5

    # 速度大小
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. 接近目标奖励：鼓励向目标移动
    # 使用负距离作为奖励，距离越近奖励越大
    distance_reward = -distance * 0.1

    # 2. 速度控制奖励：鼓励在接近目标时减速
    # 当距离较远时允许较高速度，距离近时惩罚高速
    speed_penalty = -speed * 0.05 * (1.0 / (distance + 0.5))

    # 3. 姿态稳定奖励：鼓励保持直立姿态（角度接近0）
    # 角度偏离越大惩罚越大
    angle_penalty = -abs(angle) * 0.2

    # 4. 角速度惩罚：鼓励减少旋转
    ang_vel_penalty = -abs(ang_vel) * 0.1

    # 5. 着陆奖励：当两个支撑点都接触时给予奖励
    # 同时检查是否接近目标且速度较小
    both_contact = left_contact > 0.5 and right_contact > 0.5
    landing_bonus = 0.0
    if both_contact and distance < 0.5 and speed < 0.5:
        landing_bonus = 2.0

    # 6. 引擎使用惩罚：鼓励减少不必要的引擎使用
    # action 0 是无引擎，其他动作都会消耗燃料
    engine_penalty = 0.0
    if action != 0:
        engine_penalty = -0.05

    # 7. 存活奖励：每步给予微小正奖励，鼓励持续探索
    survival_bonus = 0.01

    total_reward = (distance_reward + speed_penalty + angle_penalty +
                    ang_vel_penalty + landing_bonus + engine_penalty +
                    survival_bonus)

    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus,
        "engine_penalty": engine_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components
```

# 设计说明

**任务目标理解**：这是一个2D飞行器轨迹优化任务，智能体需要从初始位置快速到达目标着陆台，同时保持稳定姿态并最小化引擎使用。

**信号选择理由**：
- 使用 `x, y` 计算距离，这是最直接的接近度度量
- 使用 `vx, vy` 计算速度，用于控制着陆时的速度
- 使用 `angle, ang_vel` 评估姿态稳定性
- 使用 `left_contact, right_contact` 检测是否成功着陆
- 使用 `action` 惩罚不必要的引擎使用

**奖励项设计理由**：
1. **距离奖励**：连续负距离信号，每步都能提供梯度，引导智能体向目标移动
2. **速度惩罚**：与距离成反比，远距离时允许高速，近距离时强制减速，避免高速撞击
3. **姿态惩罚**：直接惩罚角度偏差，鼓励保持直立姿态
4. **角速度惩罚**：抑制旋转，增加稳定性
5. **着陆奖励**：仅在同时满足接触、近距离、低速三个条件时触发，防止虚假着陆
6. **引擎惩罚**：轻微惩罚非零动作，鼓励节能
7. **存活奖励**：微小正奖励，防止智能体过早放弃