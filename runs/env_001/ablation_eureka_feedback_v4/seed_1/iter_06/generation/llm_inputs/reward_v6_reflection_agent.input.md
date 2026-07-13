# Search objective
- target_score: 200.000000
- current_score: 33.646458
- gap_to_target: 166.353542
- target_achievement_ratio: 16.823%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 33.646458）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5

    proximity_reward = -0.1 * (x**2 + y**2)

    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    contact_proximity = next_lc * next_rc * (1.0 / (1.0 + 5.0 * dist))

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_proximity

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_proximity': contact_proximity
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=33.646458, len=850.950000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-66.450193, 292.091603]

## Reward component values (mean per episode)
- proximity_reward: -12.499653
- contact_proximity: 8.030072
- orientation_penalty: -4.318368
- velocity_penalty: -1.591719

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该环境是一个 2D 飞行器／着陆器的轨迹优化任务。主体从视口顶部中央附近起始，并受到一个随机初始力。主要目标是**以最短时间**将主体移动到视口中心的着陆平台上，并**稳定停靠**（即速度归零、姿态平稳、支撑接触）。次要目标是**尽可能少用引擎推力**，以节省燃料。智能体需要学会：
- 朝目标移动（x,y 位置逼近 0）；
- 在接近目标时减速；
- 保持竖直姿态（body_angle ≈ 0）；
- 最终让左、右支撑腿同时接触目标平台（left_support_contact=1 且 right_support_contact=1）且主体静止。

**不要混淆**：仅快速到达而不稳定停靠不算成功；仅节省燃料而不及时到达也不算成功。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（接触标志为 0.0/1.0）
- obs[0]: x_position，相对于目标着陆平台中心的水平坐标，reward_usable: true
- obs[1]: y_position，相对于着陆平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无推力，主体自由漂移
- action 1: left_orientation_engine，启动一个姿态引擎（产生偏转力矩和微小推力）
- action 2: main_engine，启动主引擎（产生沿机体方向的推力）
- action 3: right_orientation_engine，启动另一侧姿态引擎（相反偏转力矩）

注意：动作是离散的，每个时间步只能选择一种引擎或不做任何操作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能表示主体已静止并停靠在平台上（左右支撑接触均为 1，速度很小，姿态稳定）。这很可能就是成功到达并停靠的结局。
- failure-like termination: crash_or_body_contact（碰撞或不当身体接触）、horizontal_position_outside_viewport（水平位置超出视口范围）
- ambiguous termination: body_not_awake_or_settled 在没有支撑接触时也可能因能量耗尽而“沉睡”，此时可视为失败；但由于环境没有提供更细的 info，我们只能依据接触和位置判断。
- truncation: 无明确时间截断，但可以限制最大步数，本环境未说明。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）
- explicit_failure_flag_available: false（无 failure 字段）
- allowed_info_fields: {} （空字典，无可用的额外字段）
- forbidden_or_uncertain_info_fields: 所有未在 step 源码中出现的字段，例如 "success", "failure", "reason" 等均不可用。终止信号仅由 terminated 布尔值给出，且 masked step source 未提供终止原因分离信息，因此我们不能直接依赖 terminated 的标签来区分成功/失败。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（均相对于着陆平台）
- velocity: x_velocity, y_velocity
- orientation: body_angle, angular_velocity
- contact: left_support_contact, right_support_contact
- action/engine: 当前动作（可用来惩罚引擎使用）
- other: 从 obs 可计算的状态量（如到目标的距离、速度模、角度绝对值等）

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 39.50 | 39.50 | 0.00 | 713.90 | contact_bonus=2.672 orientation_penalty=-0.020 proximity_reward=-0.073 velocity_penalty=-0.017 | new_best |
| 2 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 170.40 | 170.40 | 0.00 | 471.50 | contact_bonus=0.512 orientation_penalty=-0.018 proximity_reward=-0.057 velocity_penalty=-0.012 | new_best |
| 3 | contact_reward + orientation_penalty + proximity_reward + velocity_penalty | -9.34 | 170.40 | -179.74 | 1000.00 | contact_reward=0.331 orientation_penalty=-0.019 proximity_reward=-0.063 velocity_penalty=-0.015 | no_meaningful_improvement |
| 4 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 109.84 | 170.40 | -60.55 | 842.75 | contact_bonus=0.030 orientation_penalty=-0.014 proximity_reward=-0.076 velocity_penalty=-0.019 | no_meaningful_improvement |
| 5 | contact_proximity + orientation_penalty + proximity_reward + velocity_penalty | 33.65 | 170.40 | -136.75 | 850.95 | contact_proximity=0.207 orientation_penalty=-0.016 proximity_reward=-0.063 velocity_penalty=-0.016 | unsolved_high_achievement_continue_from_best |
