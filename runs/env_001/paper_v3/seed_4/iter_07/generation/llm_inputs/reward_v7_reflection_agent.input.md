# Search objective
- target_score: 200.000000
- current_score: 4.293878
- gap_to_target: 195.706122
- target_achievement_ratio: 2.147%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 4.293878）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from observations
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Proximity to target (assumed at origin)
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Shared stability factors reused across components
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)       # bounded [0,1], 1 when still
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))  # bounded [0,1]

    # Descent quality: now contact-gated to prevent hovering exploitation.
    # Without contact, only 5% of the quality is granted, creating a ~20x
    # incentive to touch down. With any contact, full quality is awarded.
    height_factor = 1.0 / (1.0 + abs(y))        # peaks at y=0
    contact_sum = left_contact + right_contact  # 0, 1, or 2
    contact_gate = 0.05 + 0.95 * min(1.0, contact_sum)  # 0.05 when no contact, 1.0 with contact
    descent_quality = contact_gate * height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Sustained contact quality: rewards stable, settled contact with both feet
    both_contact = (left_contact + right_contact) >= 1.5
    if both_contact:
        contact_quality = factor_vel * factor_angle   # bounded [0,1], high when stable
        w_contact = 5.0
        comp_contact = w_contact * contact_quality
    else:
        comp_contact = 0.0

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total_reward = comp_prox + comp_descent + comp_contact + vel_pen + att_pen

    reward_components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'contact_quality': comp_contact,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total_reward), reward_components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the next observation
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Proximity to target (assumed at origin)
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Descent quality: continuous reward for being near ground level
    # with low speed and small attitude deviations
    # Replaces the previous contact-gated soft_landing_bonus that never fired
    height_factor = 1.0 / (1.0 + abs(y))    # peaks at y=0 (platform surface)
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    descent_quality = height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_descent + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=4.293878, len=949.850000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-52.570470, 136.300510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 694.202770 | 78.4% | 78.4% | 100.0% |
| descent_quality | 138.722873 | 15.7% | 15.7% | 100.0% |
| contact_quality | 52.608444 | 5.9% | 5.9% | 1.5% |
| velocity_penalty | -0.238645 | -0.0% | 0.0% | 99.8% |
| attitude_penalty | -0.072776 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该匿名环境是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近出发，初始带有随机作用力。任务核心目标是使飞行器尽快到达并稳定停靠在中央目标平台上，同时尽可能少地使用引擎推力。智能体需要在接近目标的过程中降低速度、保持姿态稳定，并以低速度、小角度实现双腿安全触地。次要目标是节省燃料，避免不必要的引擎动作。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 推断为 float32（未明确给出，但通常如此）
- 各维度含义：
  - obs[0]: x_position，飞行器相对于目标垫的水平坐标，reward_usable: true
  - obs[1]: y_position，飞行器相对于垫面高度的垂直坐标，reward_usable: true
  - obs[2]: x_velocity，水平线速度，reward_usable: true
  - obs[3]: y_velocity，垂直线速度，reward_usable: true
  - obs[4]: body_angle，机体姿态角，reward_usable: true
  - obs[5]: angular_velocity，角速度，reward_usable: true
  - obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
  - obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine，不做任何推进
  - action 1: left_orientation_engine，点燃左侧姿态引擎
  - action 2: main_engine，点燃主引擎
  - action 3: right_orientation_engine，点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- **body_not_awake_or_settled**：飞行器因稳定停靠而进入休眠或被判定为已安定，环境终止。此条件是**可能的成功终止**，但需要根据终止时的状态（位置接近目标、双腿接触、低速度、小角度）进一步判别。
- **crash_or_body_contact**：飞行器与地面或障碍物发生剧烈碰撞（可能倾覆或过猛接触），视为**失败终止**。
- **horizontal_position_outside_viewport**：飞行器水平飞出可视区域，视为**失败终止**。

（注：未给出 truncation 条件，无最大步数截断信息，但可以预见训练中可能设置时间上限，但环境源代码中未体现。）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 空（step 返回 info 为 {}，无可用的额外字段）
- forbidden_or_uncertain_info_fields: 禁止使用任何 info 字段，因为均为空。

成功与否必须完全从观测序列和终止时的状态（next_obs）中推断。

## 7. 可用于奖励函数的信号
- **位置信号**：obs[0] (x_position), obs[1] (y_position)；next_obs 对应维度。
- **速度信号**：obs[2] (x_velocity), obs[3] (y_velocity)；next_obs 对应维度。
- **姿态信号**：obs[4] (body_angle), obs[5] (angular_velocity)；next_obs 对应维度。
- **接触信号**：obs[6] (left_support_contact), obs[7] (right_support_contact)；next_obs 对应维度。
- **动作/引擎使用信号**：action 值本身（0 为无推力，其他为使用引擎）。
- **其他**：终止信号 `terminated` 未直接进入 reward 函数，但可通过 `next_obs` 后是否结束来间接感知（训练循环外部，通常 reward 函数不接收 terminated 标志）。若环境提供 `training_progress`，可用于调度课程。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete (4 actions)
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: impulse_based_main_and_orientation_engines
  contact_structure: left_and_right_leg_contact_sensors
primary_objectives:
  - 尽快到达目标平台中央并稳定停靠
  - 实现安全软着陆（低速度、小角度、双腿触地）
secondary_objectives:
  - 最小化引擎使用（省油）
main_failure_risks:
  - 飞行器硬撞击地面或倾覆
  - 水平漂出视野范围
  - 未能及时减速导致越标或错过平台
  - 姿态角过大导致触地后倾翻
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_reaching_and_safe_contact**
  - purpose: 激励飞行器靠近目标平台并最终在满足成功条件时稳定停靠。
  - why_required: 这是任务的核心目标，无此奖励则学习无方向。
  - usable_signals: x_position, y_position, left_support_contact, right_support_contact, velocity/angle（用于判定成功状态）。
  - risks: 若只考虑位置而不考虑速度/接触，可能导致高速撞击；需与着陆质量奖励配合。

- **role_id: soft_landing_quality**
  - purpose: 惩罚触地时过大的速度、过大的姿态角以及角速度，鼓励双腿同时接触。
  - why_required: 保证安全着陆，避免硬着陆和倾翻，是任务的关键质量要求。