# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个2D飞行器从视口顶部中央区域出发，精准降落到中央目标垫上，并保持稳定、安全的接触（两条支撑腿同时着垫、机体姿势竖直、速度极低）。  
次目标：在确保安全着陆的前提下，尽可能缩短总耗时并减少引擎燃料消耗。  
不应混淆的目标：不要单纯追求最快到达而牺牲着陆安全性；也不要只追求零燃料而无限盘旋。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达一个明确的空间目标点（目标垫），并满足姿态与接触的约束；推进、能源、时间都属于附属优化目标，不构成冲突程度相当的多目标。动力学子类型为“goal_approach_and_soft_contact”。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（或 float32，但观察含浮点标志位）
- obs[0]: x_position, 飞行器相对于目标垫的水平坐标 (m)，reward_usable: true
- obs[1]: y_position, 飞行器相对于垫面高度的垂直坐标 (m)，reward_usable: true
- obs[2]: x_velocity, 水平线速度 (m/s)，reward_usable: true
- obs[3]: y_velocity, 垂直线速度 (m/s)，reward_usable: true
- obs[4]: body_angle, 机体偏转角 (rad)，reward_usable: true
- obs[5]: angular_velocity, 角速度 (rad/s)，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志 (0.0/1.0)，reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志 (0.0/1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: do nothing (no_engine) – 不点火，仅靠惯性运动
- action 1: fire left orientation engine – 产生顺时针力矩，改变姿态
- action 2: fire main engine – 产生向上的推力（同时可能产生微小水平分力？此处按垂直主推力理解）
- action 3: fire right orientation engine – 产生逆时针力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 两腿同时接触目标垫且机体姿态、速度满足稳定条件 → 触发 body_not_awake_or_settled，可能伴随低速度/低角速度，判为成功。
- failure-like termination: ① 机体任何部分与地面或非垫区域接触 (crash_or_body_contact)；② 水平位置超出视口边界 (horizontal_position_outside_viewport)；③ 在未安全着陆的情况下触发休眠 (body_not_awake_or_settled) 也可能失败。
- ambiguous termination: 当 body_not_awake_or_settled 的触发阈值较宽时，可能涵盖成功着陆（两腿着垫、速度低）与卡在边缘/侧翻但速度为零的情况，需由观察进一步区分。
- truncation: 未观察到显式截断，False 表示不使用截断，episode 总是由上述条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} (info 为空字典，不可使用)
- forbidden_or_uncertain_info_fields: 任何 info 字段均禁止使用（因为未声明）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- obs：当前观察，可读所有 8 维
- action：当前执行的动作（此步的输入 action）
- next_obs：下一时刻观察（注意：环境 step 返回的是 next state 作为输出，但接口约定传递的是 transition 元素。此处考虑常见设计，next_obs 可用）
- info：空字典，不可用任何字段

禁止使用：
- original_reward：已被屏蔽，不可访问
- official_reward：不可使用
- training_progress：除非 prompt 明确允许，否则不使用
- 任何未声明的 info 字段

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: current action (动作索引或 one-hot)，可计算是否使用主引擎/姿态引擎
- other: 无

## 8. 不确定或不可用的信号
- 成功/失败标志不可用
- 剩余燃料量、时间步等内部状态未暴露
- 目标垫位置绝对值不可知（只有相对位置）
- 机体质量、惯性等物理参数未知
- 风力等随机扰动未暴露

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: discrete_thrusters (main + two orientation engines)
  contact_structure: two independent leg contacts (left, right)
primary_objectives:
  - land on target pad with both legs touching and body upright
  - keep low linear and angular velocity at touchdown
secondary_objectives:
  - minimize fuel consumption
  - reach landing quickly (time efficiency)
main_failure_risks:
  - crashing into ground or obstacles
  - moving out of horizontal bounds
  - landing with only one leg, causing tilt and subsequent crash
  - overshooting target and drifting away
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_approach
  - purpose: 鼓励飞行器接近目标垫，降低相对位置误差；特别是在水平方向防止漂移。
  - why_required: 核心导航目标，若没有该项，智能体可能不会向目标移动。
  - usable_signals: [x_position, y_position]
  - risks: 如果只奖励位置接近，可能鼓励高速撞击，需要配合速度抑制。

- role_id: soft_landing
  - purpose: 在接近垫面且具备接触条件时，要求低线速度、低角速度，确保安全着陆。
  - why_required: 仅有位置到达不够，接触时的大速度会导致坠毁或弹起，必须抑制。
  - usable_signals: [x_velocity, y_velocity, angular_velocity, left_support_contact, right_support_contact]
  - risks: 过早强调速度约束可能抑制必要机动，应与高度条件耦合。

- role_id: crash_avoidance
  - purpose: 对机体碰撞（地面、边界）施加严厉惩罚，防止不安全行为。
  - why_required: 终止条件包含 crash_or_body_contact 和越界，这些失败需要从奖励中学习避免。
  - usable_signals: 下一时刻的终止标志 (terminated) 可通过函数外部推断（如果允许传递 done 标志，但接口未显式包含，只能通过判断 next_obs 的某些异常模式，如位置越界突跳）。更稳妥的方式：从 obs 直接检测风险状态，如 y_position 过低且无接触，或水平位置接近边界，或两腿接触地面但 body_angle 过大等。usable_signals: [x_position, y_position, body_angle, left_support_contact, right_support_contact] 用于构造启发式惩罚。
  - risks: 若仅依赖终止标志（无法直接获得）则不可靠；应使用 pre-terminal 危险状态检测。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  - condition_to_use: 仅在训练后期或着陆阶段（如高度低于某阈值）加入，以避免早期干扰探索。
  - usable_signals: [action == 2, action == 1 or 3?] 可计算主引擎使用惩罚。
  - risks: 过早加入会抑制智能体学习必要的推力机动，导致无法到达目标。

- role_id: orientation_penalty
  - condition_to_use: 当接近地面时（y_position 较小），要求姿态竖直，否则强惩罚。
  - usable_signals: [body_angle, y_position, contact flags]
  - risks: 在远距离不必要，可能限制姿态调整；过早使用可能造成局部最优。

- role_id: progress_time_bonus
  - condition_to_use: 仅当安全着陆趋势明显时（例如高度下降且速度可控），给予微弱的时间奖励；否则坚决不使用，以防止冒险行为。
  - usable_signals: [y_position, y_velocity, contact]
  - risks: 极易导致高速坠落，必须严格控制条件。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: dense_speed_penalty (generic)
  - reason: 全时间对速度施加平方惩罚会与接近目标所需的机动矛盾，导致低效探索。
  - forbidden_or_missing_signals: 需要精细速度阈值，当前无任务阶段信息。

- role_id: pure_sparse_success_reward
  - reason: 无明确成功标志；仅终止时基于接触和速度手工判定成功不可靠，容易引入噪声，且稀疏奖励会使训练极难。
  - forbidden_or_missing_signals: success flag 缺失。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_approach | x_position, y_position | target absolute coordinate (relative is fine) | distance_to_goal (L2), negative position penalty, shaping: decrease/increase with distance | 注意不要激活碰撞时奖励 |
| soft_landing | y_velocity, x_velocity, angular_velocity, left_contact, right_contact, y_position | terminal velocity threshold | conditional penalty: `if close_to_ground: -‖v‖² - ‖ω‖²` | 需要与高度或接触耦合，避免全局抑制 |
| crash_avoidance | x_position, y_position, body_angle, contact | true crash indicator | harsh negative constant on crash-like state (e.g., y<0 & no contact, |x|>limit, etc.) | 使用环境知识构造 risk zone |
| fuel_efficiency | current_action (main engine) | fuel gauge | small negative reward for action==2, possibly scaled down if close to ground | 条件激活，与训练进度相关 |
| orientation_penalty | body_angle, y_position, contacts | ground normal | `-|sin(angle)|` when near ground | 避免在远距离干扰 |
| progress_time_bonus | y_position, y_velocity | elapsed time | negligible bonus only under strong safety condition | 高风险禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器悬停在高处不下降 | 平均 episode 回报停滞，y_position 不离零附近 | 增加下降激励（例如 y_position 的变化奖励），或放宽速度惩罚 |
| 高速撞击目标垫导致 crash | 高速度、终止时 y_position≈0 但 contact 不全 | 加强 soft_landing 职责，特别是接近垫面时的速度惩罚 |
| 持续向一侧漂移超出边界 | x_position 单调增加/减少至边界 | 增加横向位置代价，或引入水平方向稳定奖励 |
| 着陆时仅单腿接触并侧翻 | left_contact xor right_contact 且 body_angle 大 | 增加对双接触的一致性奖励，并加强姿态惩罚 |
| 过度使用主引擎燃料消耗极大 | 动作序列主要由 action 2 组成，燃料无限制（但燃料不可见，可观测行为） | 增加燃料效率条件职责，限制主引擎连续使用次数（可通过动作历史实现） |
| 无法学习有效着陆，始终在垫附近震荡 | 位置误差小但速度控制不稳定，终止时间过长 | 考虑加入基于 y_position 的进度塑造，或调整 soft_landing 权重与衰减条件 |