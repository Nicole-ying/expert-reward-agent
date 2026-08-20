# Search objective
- target_score: 200.000000
- current_score: -30.778019
- gap_to_target: 230.778019
- target_achievement_ratio: -15.389%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -30.778019）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: geometric mean of 3 continuous factors (proximity, low speed,
      upright posture) providing dense approach gradient; plus additive contact bonus
      scaled by approach quality to incentivize final touchdown without collapsing
      the approach signal
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: landing proxy with decoupled approach quality and contact bonus
    # Bounded factors for continuous dimensions
    prox_factor = max(0.0, 1.0 - dist / 2.5)
    speed = (vx**2 + vy**2) ** 0.5
    vel_factor = max(0.0, 1.0 - speed / 2.0)
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # Geometric mean of 3 continuous factors: enforces joint satisfaction
    # of proximity, low speed, and upright posture during approach.
    # Unlike the 4-factor version, this does NOT collapse when airborne
    # (contact_factor was the zero-killer for 94% of steps).
    approach_quality = (prox_factor * vel_factor * angle_factor) ** (1.0 / 3.0)

    # Contact factor: 0.0 (none), 0.5 (single), 1.0 (double)
    contact_factor = (left_contact + right_contact) / 2.0

    # Contact bonus: additive extra reward when touching ground,
    # scaled by approach_quality so contact only pays off when
    # the vehicle is also close, slow, and upright.
    contact_bonus = contact_factor * approach_quality

    w_approach = 5.0
    w_contact = 3.0
    landing_proxy = w_approach * approach_quality + w_contact * contact_bonus

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: discourages high speed when close to the target
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: sum of bounded factors rewarding simultaneous proximity,
      low speed, upright posture, and ground contact
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)          # near target -> gate approx 1; far away -> gate approx 0
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: soft landing proxy (sum of bounded factors, non-collapsing joint)
    w_landing = 3.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_reward = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_reward = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad (~28.6 deg)
    angle_reward = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0, 0.5, or 1.0 depending on single/double contact
    contact_reward = (left_contact + right_contact) / 2.0
    # sum of four independent bounded factors, averaged
    landing_proxy = w_landing * (prox_reward + vel_reward + angle_reward + contact_reward) / 4.0

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-30.778019, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-57.297976, 6.039358]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 4403.989123 | 88.2% | 88.2% | 100.0% |
| distance_penalty | -580.923112 | -11.6% | 11.6% | 100.0% |
| orientation_penalty | -6.370182 | -0.1% | 0.1% | 100.0% |
| velocity_penalty | -2.611327 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体（近似2D车辆）从视口顶部中心附近以随机初始推力出发，必须尽快到达中央目标平台并稳定停靠（settle），同时尽可能少用引擎推力。智能体需要学会规划轨迹、减速、保持姿态平稳，并在安全接触（双支撑点触地）的条件下结束。任务主目标为“到达并稳定停靠在目标平台”；次目标为“最小化燃料消耗”和“尽快完成”；不应与持续运动、抓取或空地跟踪混淆。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: 推断为 float32（未显式指定，观察字段为位置/速度/角度等连续标量）
- obs[0]: x_position，目标平台的水平相对坐标（m），reward_usable: true
- obs[1]: y_position，相对目标平台高度的垂直坐标（m），reward_usable: true
- obs[2]: x_velocity，水平线速度（m/s），reward_usable: true
- obs[3]: y_velocity，垂直线速度（m/s），reward_usable: true
- obs[4]: body_angle，身体的朝向角（rad），reward_usable: true
- obs[5]: angular_velocity，角速度（rad/s），reward_usable: true
- obs[6]: left_support_contact，左支撑接触标志（0.0/1.0），reward_usable: true
- obs[7]: right_support_contact，右支撑接触标志（0.0/1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine – 无推力/转矩，仅靠已有动量演化
- action 1: left_orientation_engine – 激发左侧方向引擎（推测产生逆时针/正向转矩，用于调节角度）
- action 2: main_engine – 激发主引擎（推测沿体轴方向产生线性推力，用于加速前进）
- action 3: right_orientation_engine – 激发右侧方向引擎（推测产生顺时针/负向转矩，与动作1相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 可能表示身体停止运动（速度、角速度降至阈值以下），如果同时满足接近目标、双支撑接触、姿态接近水平，则很可能为成功着陆。
- failure-like termination: `crash_or_body_contact` 表示身体与其他物体（地面、墙壁）发生不良碰撞；`horizontal_position_outside_viewport` 表示水平位置超出视野，通常视为任务失败。
- ambiguous termination: `body_not_awake_or_settled` 也可能在偏离目标或接触不良时触发，此时不能直接认定为成功，需结合 obs 进一步判断。
- truncation: 未在 step 中显式出现（info 为空，无 `TimeLimit.truncated`），但环境可能自带最大步数限制，不应依赖外部截断信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 固定为 `{}`，禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；禁止依赖 `info["success"]`、`info["failure"]` 或任何终止原因字符串。

## 7. 可用于奖励函数的信号
- position: obs[0] x_position, obs[1] y_position（均相对于目标）
- velocity: obs[2] x_velocity, obs[3] y_velocity
- orientation: obs[4] body_angle, obs[5] angular_velocity
- contact: obs[6] left_support_contact, obs[7] right_support_contact（布尔标志形式）
- action: action 索引（0~3），可用于鼓励节能或惩罚特定动作
- other: 可通过 next_obs 计算增量（例如速度变化），或判断 settled 条件（速度绝对值 < ε，角度 < ε'，双接触为真等）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 近似2D小车，带刚性车体
  actuator_type: 一个主推进引擎（向前），两个方向引擎（左右，各产生转向力矩）
  contact_structure: 两个底部支撑点（左右），接触信息以布尔标志提供
primary_objectives:
  - 精确水平定位至目标（x_position → 0）
  - 高度匹配平台（y_position → 0）
  - 稳定停靠（线速度/角速度 → 0，双支撑接触 = 1）
secondary_objectives:
  - 最小化引擎使用次数（节能）
  - 尽快完成，鼓励高效轨迹
main_failure_risks:
  - 过高速度或角度导致硬着陆（触地碰撞）
  - 偏离目标区域导致 settled 在不合法位置
  - 长时间悬停或重复无用动作浪费时间步
  - 姿态过分倾斜导致翻滚或接触不良
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: position_approaching
  purpose: 鼓励缩小与目标的水平/垂直距离
  why_required: 任务核心是到达，缺失将使智能体无方向指引
  usable_signals: [obs[0] x_position, obs[1] y_position, next_obs[0], next_obs[1]]
  risks: 过度奖励可能产生振荡接近而非稳定着陆；需与速度阻尼结合

- role_id: velocity_reduction
  purpose: 靠近目标时抑制速度，防止高速撞击
  why_required: 安全接触必须低动能，否则触发 crash
  usable_signals: [obs[2] x_velocity, obs[3] y_velocity, next_obs[2], next_obs[3]]
  risks: 过早减速会降低效率；最好根据与目标距离动态加权（近目标时惩罚大）

- role_id: orientation_stabilization
  purpose: 保持车体水平（角度接近0），避免翻滚或接触不良
  why_required: 稳定着陆需双支撑同时触地，倾斜过大导致单侧接触或碰撞
  usable_signals: [obs[4] body_angle, obs[5] angular_velocity]
  risks: 过度限制角度可能阻碍机动；可在接近目标后增强权重

- role_id: safe_landing_confirmation
  purpose: 鼓励双支撑接触同时所有运动停止时给予成功奖励
  why_required: 明确任务完成的定义，并提供稀疏奖励信号
  usable_signals: [next_obs[2..5] 速度/角速度，next_obs[6] left_contact, next_obs[7] right_contact]
  risks: 必须严格检查 settled 条件，否则可能给虚假成功；需防过早 settled 在错误位置

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  condition_to_use: 当环境步数预算紧张或希望显式节能时启用；也可以始终实行为弱正则化
  usable_signals: [action, 通过动作索引区分是否使用引擎]
  risks: 权重过大会阻止主动机动，导致智能体“躺平”；建议作为小量级惩罚

- role_id: time_efficiency_encouragement
  condition_to_use: 如果希望显式加速完成，可通过微小每步消耗间接实现；此处因无可靠步数计数器，可通过鼓励快速到目标替代，但需靠稀疏成功奖励自然驱动
  usable_signals: 无直接信号；可使用 training_progress（未启用，禁用）
  risks: 引入不可靠的时间压力可能导致危险动作

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_force_smoothness
  reason: 环境仅提供布尔接触标志，无接触力信息，无法实现基于力传感器的高级接触奖励
  forbidden_or_missing_signals: [contact_force]

- role_id: explicit_progress_race
  reason: 缺乏分段进度里程碑或路线进度指标，无法定义沿轨迹的进度，避免过度复杂报酬设计
  forbidden_or_missing_signals: [absolute_x_distance_progress]

- role_id: absolute_time_penalty
  reason: 未暴露时间片信息，强行使用 training_progress 违反接口契约，禁用
  forbidden_or_missing_signals: [step_count, elapsed_time]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| position_approaching | x_position, y_position | 无 | -distance_to_target, dense_state_signal | 可用平方距离，可分离水平和垂直 |
| velocity_reduction | x_velocity, y_velocity | 无 | -quadratic_penalty_on_velocity, bounded_signal | 建议乘以距离衰减因子 |
| orientation_stabilization | body_angle, angular_velocity | 无 | -abs(angle) - scaled angular_velocity | 角度可采用绝对值，角速度抑制 |
| safe_landing_confirmation | x_velocity, y_velocity, angular_velocity, left_contact, right_contact | 无 | sparse_bonus_on_check(settled & on_target & double_contact) | 必须结合位置阈值和接触条件 |
| fuel_efficiency_penalty | action_index | 无 | -small_constant_if(action in {1,2,3}) | 离散惩罚或按动作计数 |
| time_efficiency_encouragement | 无 | step_counter | (disabled) | 可通过成功速度快回归到节能，不单独设计 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过早 settled（在远离目标的位置） | next_obs 满足 settled 但 x_position, y_position 较大 | 收紧 settled 的位置阈值；增加距离相关的惩罚 |
| 高速撞击地面（crash） | 终止前速度大