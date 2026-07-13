# Search objective
- target_score: 200.000000
- current_score: -138.372823
- gap_to_target: 338.372823
- target_achievement_ratio: -69.186%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -138.372823）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # 进步奖励：向目标靠近
    progress = prev_dist - dist
    progress_reward = 2.0 * progress

    # 速度惩罚：高速时惩罚，接近目标时放大
    vel_gate = 1.0 / (1.0 + 3.0 * dist)
    velocity_penalty = -5.0 * (speed**2) * vel_gate

    # 姿态惩罚
    orientation_penalty = -0.1 * (angle**2) - 0.05 * (av**2)

    # 接触奖励：双接触 + 低速 + 竖直
    contact_bonus = 10.0 * lc * rc * max(0.0, 1.0 - speed) * max(0.0, 1.0 - abs(angle))

    total = progress_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'progress_reward': progress_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=-138.372823, len=668.300000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-213.278637, -47.881227]

## Reward component values (mean per episode)
- velocity_penalty: -34.067878
- progress_reward: 0.611853
- orientation_penalty: -0.248246
- contact_bonus: 0.000000

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
| 6 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 34.67 | 170.40 | -135.73 | 321.25 | contact_bonus=5.363 orientation_penalty=-0.019 proximity_reward=-0.069 velocity_penalty=-0.014 | no_meaningful_improvement |
| 7 | orientation_penalty + proximity_reward + success_reward + velocity_penalty | -115.04 | 170.40 | -285.44 | 68.45 | orientation_penalty=-0.031 proximity_reward=-0.486 success_reward=0.067 velocity_penalty=-0.037 | no_meaningful_improvement |
| 8 | contact_bonus + orientation_penalty + progress_reward + velocity_penalty | -138.37 | 170.40 | -308.77 | 668.30 | contact_bonus=3.203 orientation_penalty=-0.010 progress_reward=0.006 velocity_penalty=-0.331 | unsolved_high_achievement_continue_from_best |
