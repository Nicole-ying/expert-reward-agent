# Search objective
- target_score: 200.000000
- current_score: 71.060291
- gap_to_target: 128.939709
- target_achievement_ratio: 35.530%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 71.060291）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前与下一状态
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new, vy_new = next_obs[2], next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 到原点的距离（目标平台位于 (0,0)）
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5

    # 权重
    w_progress = 5.0
    w_y_progress = 2.0
    w_alignment = 0.5
    w_horizontal_penalty = 0.05
    w_vel = 0.1
    w_att = 0.1
    w_contact = 3.0
    w_ground = 1.0

    # 阈值
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2
    ideal_angle_tol = 0.1
    ideal_vy_tol = 0.2

    # 1. 向目标移动的稠密进展奖励（距离缩短）
    progress = w_progress * (dist_old - dist_new)

    # 2. 垂直下降奖励（直接奖励高度减小）
    y_progress = w_y_progress * (y_old - y_new)

    # 3. 水平对准奖励（接近 x=0 时高奖励）
    alignment = w_alignment * (1.0 / (1.0 + x_new ** 2))

    # 4. 微小水平偏离惩罚（仅在需要时提供软边界）
    horizontal_penalty = -w_horizontal_penalty * (x_new ** 2)

    # 5. 安全速度约束
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx ** 2 + excess_vy ** 2)

    # 6. 姿态安全约束
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle ** 2 + excess_angvel ** 2)

    # 7. 接触奖励（任意支撑腿接触 + 质量因子）
    any_contact = max(left_contact, right_contact)
    angle_quality = max(0.0, 1.0 - abs(angle_new) / ideal_angle_tol)
    vy_quality = max(0.0, 1.0 - abs(vy_new) / ideal_vy_tol)
    contact_reward = w_contact * any_contact * angle_quality * vy_quality

    # 8. 接近地面奖励（当高度低于阈值时线性激励）
    ground_threshold = 0.5
    height_near_reward = w_ground * max(0.0, ground_threshold - y_new)

    # 总和
    total_reward = (progress + y_progress + alignment + horizontal_penalty +
                    vel_penalty + att_penalty + contact_reward + height_near_reward)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'alignment': alignment,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_reward': contact_reward,
        'height_near_reward': height_near_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=71.060291, len=977.600000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[17.969578, 176.793787]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 633.930169 | 42.8% | 42.8% | 26.0% |
| alignment | 479.401371 | 32.3% | 32.3% | 100.0% |
| height_near_reward | 356.299293 | 24.0% | 24.0% | 85.7% |
| progress | 6.293330 | 0.4% | 0.6% | 100.0% |
| y_progress | 2.818477 | 0.2% | 0.2% | 93.1% |
| horizontal_penalty | -1.001960 | -0.1% | 0.1% | 100.0% |
| vel_penalty | -0.023368 | -0.0% | 0.0% | 1.4% |
| att_penalty | -0.014444 | -0.0% | 0.0% | 2.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务：在二维平面中，控制一个带有两侧支撑腿的着陆器从视野上方中心附近受随机初速出发，**安全、稳定地降落到视野中央的目标平台上**，要求着陆器最终静止、姿态竖直且两侧支撑腿均与平台接触。  
次任务：在保证主目标的前提下，**尽可能快地完成着陆**，并且**尽量降低发动机推力消耗**（即少用燃料）。  
不应混淆的目标：燃料优化与快速着陆是辅助目标，它们不能凌驾于安全稳定着陆之上；本任务的最终评判是“是否成功停靠在平台上”这一离散事件，并非将能耗作为等价的主要目标。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: 根据上下文推断为 float32 (或混合了 bool 转为 float 的标志位，实际为连续量)
- **各维度含义**：
  - `obs[0]`: x_position (目标平台的相对水平坐标) —— reward_usable: true，用于控制水平对准。
  - `obs[1]`: y_position (相对于目标平台高度的垂直坐标) —— reward_usable: true，用于测量剩余高度/下降进度。
  - `obs[2]`: x_velocity (水平线速度) —— reward_usable: true，用于减速控制。
  - `obs[3]`: y_velocity (垂直线速度) —— reward_usable: true，用于控制坠落/着陆速度。
  - `obs[4]`: body_angle (机体偏转角) —— reward_usable: true，用于保持竖直姿态。
  - `obs[5]`: angular_velocity (角速度) —— reward_usable: true，姿态稳定惩罚。
  - `obs[6]`: left_support_contact (左支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。
  - `obs[7]`: right_support_contact (右支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `action 0` (no_engine): 不启动任何引擎（零推力）。
  - `action 1` (left_orientation_engine): 启动左侧姿态引擎，产生旋转力矩。
  - `action 2` (main_engine): 启动主引擎，产生向上的升力。
  - `action 3` (right_orientation_engine): 启动右侧姿态引擎，产生相反的旋转力矩。
- **值域**：`[0, 1, 2, 3]`，每个动作只消耗一个离散选择的“燃料单位”。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 机体静止且（可能）已与平台产生接触。根据任务描述“settle at a central target pad... make safe contact”，此条件很可能标志着成功着陆。但是否伴随 `left_support_contact` 和 `right_support_contact` 同时为真，需要观察，但终止代码中只用此单一条件判定终止，可能表示“已稳定着陆”。
- **failure-like termination**:  
  `crash_or_body_contact` —— 机体与外环境发生碰撞（非支撑腿安全着陆），通常是坠毁。  
  `horizontal_position_outside_viewport` —— 水平飘出视野边界，失败。
- **ambiguous termination**:  
  在没有显式成功标志的情况下，终止由这三种条件之一触发，其中 `body_not_awake_or_settled` 语义为“非清醒或已稳定”，我们暂时将其视为**成功候选**，但无法 100% 确认。需要后续根据经验收集证据。
- **truncation**:  
  源代码未展示任何步数限制或超时截断（`return ..., False, {}` 中没有 truncated）。如果存在，可能在上层包装器中，但当前分析不可见。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 字典为空）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info={} 且没有声明任何可用字段）
- **forbidden_or_uncertain_info_fields**: 所有 info 字段都不允许使用，因为没有可靠信息源。`original_reward` 被屏蔽，禁止访问。

## 7. 可用于奖励函数的信号
- **position**:
  - `next_obs[0]` (x_position)：到目标平台的水平距离。
  - `next_obs[1]` (y_position)：到目标平台高度的垂直距离。
- **velocity**:
  - `next_obs[2]` (x_velocity)
  - `next_obs[3]` (y_velocity)
- **orientation**:
  - `next_obs[4]` (body_angle)
  - `next_obs[5]` (angular_velocity)
- **contact**:
  - `next_obs[6]` (left_support_contact)
  - `next_obs[7]` (right_support_contact)
- **action/engine**:
  - `action`：离散动作编号，可用于惩罚引擎使用（燃料消耗），但不能直接读出推力大小。
- **other**:  
  无。

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
| 1 | contact_establishment + goal_proximity + soft_landing_stabilization + upright_attitude | -111.84 | -111.84 | 0.00 | 68.50 | contact_establishment=0.050 goal_proximity=-1.075 soft_landing_stabilization=-0.133 upright_attitude=-0.041 | new_best |
| 2 | att_penalty + contact_bonus + horizontal_penalty + progress + vel_penalty | -21.25 | -21.25 | 0.00 | 1000.00 | att_penalty=-0.008 contact_bonus=0.864 horizontal_penalty=-0.019 progress=0.003 vel_penalty=-0.009 | new_best |
| 3 | alignment + att_penalty + contact_reward + height_near_reward + horizontal_penalty + progress | 71.06 | 71.06 | 0.00 | 977.60 | alignment=0.469 att_penalty=-0.003 contact_reward=0.868 height_near_reward=0.293 horizontal_penalty=-0.004 | new_best |
