# 匿名环境理解卡片

## 1. 任务目标
任务主目标：控制一个具有两个支撑接触点的二维飞行器（类似着陆器）从视野顶部中心附近出发，尽快到达并稳定停靠在中央目标平台上，同时尽量节省发动机燃料。代理需学会准确接近目标（水平与垂直位置归零）、降低速度、保持竖直姿态并实现双腿同时平稳触地。  
不被混淆的目标：纯粹最小化燃料消耗或最大程度悬停都不是目标，核心是“快速、安全、节能地完成停靠”。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标为到达并停靠在明确的目标位置，附属有速度、姿态、能耗等约束，无其他权重相当的冲突目标，符合导航型任务定义。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（至少部分值为浮点，接触标志为实数 0.0/1.0）
- obs[0]: x_position – 飞行器相对于目标平台的水平偏移，reward_usable: true
- obs[1]: y_position – 飞行器相对于平台高度的高度差（0表示正好在平台高度），reward_usable: true
- obs[2]: x_velocity – 水平线速度，reward_usable: true
- obs[3]: y_velocity – 垂直线速度，reward_usable: true
- obs[4]: body_angle – 飞行器俯仰角/倾斜角，reward_usable: true
- obs[5]: angular_velocity – 角速度，reward_usable: true
- obs[6]: left_support_contact – 左支撑接触标志（1.0接触，0.0脱离），reward_usable: true
- obs[7]: right_support_contact – 右支撑接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine – 关闭所有发动机，无推力
- action/action_dim 1: left_orientation_engine – 点燃左侧姿态发动机（产生旋转力矩及可能少量平移力）
- action/action_dim 2: main_engine – 点燃主发动机（主要提供垂直向上推力）
- action/action_dim 3: right_orientation_engine – 点燃右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 若发生在飞行器位于目标平台中心且接触双腿着地时，可视为成功停靠。
- failure-like termination: crash_or_body_contact（异常碰撞）、horizontal_position_outside_viewport（水平出界）、body_not_awake_or_settled 若发生在远离平台或仅单腿接触等不安全区域，也视为失败。
- ambiguous termination: 仅凭 termination 标志无法区分成功/失败，需结合最终状态判断。
- truncation: 无显式 episode 截断信息，step source 中 truncated 固定为 False。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（返回空字典 {})
- forbidden_or_uncertain_info_fields: 一切 info 字段，因无任何明确声明。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前时刻观察）
- action（当前动作）
- next_obs（下一时刻观察）
- info（当前为空，不允许假设任何字段）
- training_progress（此处 prompt 未明确允许使用，默认禁用）

禁止使用：
- original_reward（官方奖励已屏蔽）
- 任何未在 3. 节中声明的 obs 切片含义以外的信号
- 任何未从 step 源码中明确出现的 info 字段

## 7. 可用于奖励函数的信号
- position: next_obs[0] (x), next_obs[1] (y) —— 可用于计算距目标平台中心的距离或高度差
- velocity: next_obs[2] (vx), next_obs[3] (vy) —— 可用于惩罚高速着陆或水平漂移
- orientation: next_obs[4] (angle) —— 可用于鼓励竖直姿态
- angular_velocity: next_obs[5] —— 可用于惩罚剧烈旋转
- contact: next_obs[6], next_obs[7] —— 可用于奖励双腿同时接触平台
- action/engine: action（0～3） —— 可用于惩罚使用主发动机及姿态发动机，鼓励少用推力

## 8. 不确定或不可用的信号
- 无官方 success/failure 标志
- 无触地冲击力或触地速度阈值信息
- 无环境是否处于 “settled” 状态的明确中间信号（只有环境内部判断）
- 无实时燃料消耗数值，只能通过动作类别间接反映

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_landing_legs
  actuator_type: discrete_thrusters (main + two orientation)
  contact_structure: two-point_support (left/right)
primary_objectives:
  - reach_target_pad (minimize final distance to pad center, x→0, y→0)
  - achieve_stable_settle (zero relative velocity, upright, both legs contact)
  - minimize_time (implicitly via density of positive rewards)
secondary_objectives:
  - minimize_fuel_or_engine_usage
main_failure_risks:
  - crash_or_body_contact (hard landing, hitting ground/edge)
  - horizontal_out_of_bounds
  - unstable_oscillation (excessive rotation, bouncing)
  - fuel-inefficient hovering or overshoot
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target_position
  purpose: 驱使飞行器水平与垂直位置接近目标平台中心 (0,0)
  why_required: 主任务“到达目标”的直接量化
  usable_signals: [next_obs[0], next_obs[1]] → 距离/平方距离
  risks: 可能鼓励高速撞击，需与速度惩罚配合

- role_id: stabilize_orientation_and_angular_velocity
  purpose: 鼓励竖直姿态、抑制旋转
  why_required: 安全软著陆的关键前提
  usable_signals: [next_obs[4], next_obs[5]] → 角度绝对值、角速度平方
  risks: 若权重过高，可能让飞行器悬停过久

- role_id: penalize_engine_usage
  purpose: 节省燃料 / 最小化发动机使用
  why_required: 任务描述的节能要求
  usable_signals: [action] → 是否使用主发动机 / 姿态发动机
  risks: 与快速到达目标冲突，需适当平衡权重

### 10.2 条件职责 conditional_roles
- role_id: soft_landing_velocity_penalty
  condition_to_use: 当接触标志从无到有或持续接触时，惩罚过大的垂直或水平速度
  usable_signals: [next_obs[2], next_obs[3], next_obs[6], next_obs[7]] → 速度大小 × 接触标志
  risks: 延迟给予成功奖励，可能阻碍接近阶段正常速度

- role_id: dual_leg_contact_bonus
  condition_to_use: 只在接近目标位置且低速时，额外奖励双腿同时接触
  usable_signals: [next_obs[0], next_obs[1], next_obs[6], next_obs[7]] → 同时为1且位置接近0
  risks: 可能误导代理优先追求接触而忽略位置精度

### 10.3 慎用/禁用职责 avoid_roles
- role_id: early_termination_success_bonus
  reason: 无显式成功标志；仅凭 body_not_awake_or_settled 不可靠，可能奖励在半途停稳（非目标位置）的局部静止
  forbidden_or_missing_signals: [explicit success flag, allowed info fields]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target_position | next_obs[0], next_obs[1] | — | -distance_to_target, -squared_distance, bounded_signal | 可用欧氏距离或各分量平方和，离目标越近奖励越高 |
| stabilize_orientation_and_angular_velocity | next_obs[4], next_obs[5] | — | -abs(angle), -angle^2, -ang_vel^2, quadratic_penalty | 角度接近0获得最高奖励 |
| penalize_engine_usage | action (0–3) | 准确燃料消耗值 | -I(action!=0), -per_action_cost | 不同动作可设不同代价（主引擎>姿态>空闲） |
| soft_landing_velocity_penalty | next_obs[2], next_obs[3], next_obs[6], next_obs[7] | 实际接触力 | -contact * velocity_magnitude | 仅在接触时乘速度惩罚，避免正常接近阶段被罚 |
| dual_leg_contact_bonus | next_obs[6], next_obs[7], next_obs[0], next_obs[1] | — | use contact_and_position_gate | 要求同时接触且位置靠近0才给正奖励 |
| early_termination_success_bonus | — | explicit success info | avoid | 禁止使用，缺乏可靠信号 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不降或下降过慢 | y位置长时间不减小，主引擎使用率低 | 加强高度下降的奖励形状，或时间衰减调整权重 |
| 高速撞击目标平台 | 终止前瞬间速度很大，crash_or_body_contact | 提升速度惩罚权重，或增加接触时速度惩罚 |
| 单腿先触地导致倾倒 | 左右接触标志不同步，角度大幅变化后终止 | 增加双腿同步奖励，接触时必须角度接近0 |
| 水平漂移出界 | x位置超出范围前无纠正动作 | 强化水平位置接近0的奖励，或引入水平出界惩罚梯度 |
| 往复摆动或振荡 | 动作频繁切换 1/3，角速度幅度大 | 引入动作平滑代价，或加大 angular_velocity 惩罚 |
| 过度使用主发动机保安全 | 主引擎几乎全程开启，燃料理论消耗大 | 加强动作代价，或只在接近目标时允许适量使用 |