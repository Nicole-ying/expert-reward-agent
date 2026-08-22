# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主要目标：控制一个二维刚体从上方初始位置尽快到达并稳定停靠在视野中央的目标垫上。  
次要目标：在满足成功停靠的前提下，尽可能少地使用发动机推力（节约燃料/能量）。  
不允许混淆的目标：这是一个典型的到达＋软着陆任务，核心不是维持平衡或长时间存活，而是以安全、低能耗的方式精确到达指定位置。快速性隐含在“as fast as possible”中，但需要与能耗目标协调。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务描述的核心是“到达并稳定在中央目标垫”，位置目标明确且是唯一必须完成的子任务；燃料节省和快速到达只是附属优化。没有多个对等冲突目标。没有生存或探索需求。因此选择导航/到达类任务族最合适。  
dynamics_subtype: goal_approach_and_soft_contact  
说明：接近目标时需要降低速度、保持姿态、最终实现两条着陆腿同时/相继安全触地，非常接近软着陆/停靠类动力学形态。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，前6维连续，后2维取0/1但通常保持浮点）
- obs[0]: x_position，相对于目标垫中心的水平距离，reward_usable: true
- obs[1]: y_position，相对于垫面的垂直距离（高度），reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，刚体俯仰角（与垂直/水平方向的偏差），reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿触地标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿触地标志（0/1），reward_usable: true

所有8维均可直接用于奖励计算（位移、速度、姿态、接触状态）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不施加任何推力，惯性飞行
- action 1: left_orientation_engine — 点燃一侧方向引擎（可能产生侧向力矩和/或力，用于调整姿态）
- action 2: main_engine — 点燃主引擎（产生向上的推力，同时可能引起力矩变化）
- action 3: right_orientation_engine — 点燃另一侧方向引擎（与左引擎相反方向）

注意：动作空间是符号化的，但每个动作对“燃料消耗”有隐含影响（0号动作不消耗燃料）。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（“settled”指身体静止并稳定地站立在垫上）是可能的成功终止；但仅凭该条件无法绝对区分成败，因为也可能包含非成功静止（如横躺在地面但物理不活动）。
- failure-like termination: `crash_or_body_contact`（主体部分与地面或其它障碍激烈碰撞）、`horizontal_position_outside_viewport`（偏离目标太远）。
- ambiguous termination: 所有的终止条件均不显式标记成功/失败，需结合观测（位置、速度、姿态、双腿触地）判断。
- truncation: 源未提供步数截断信息，假设环境无显式步数限制，但从实际训练安全考虑通常会有时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无可用字段）
- forbidden_or_uncertain_info_fields: 没有任何 info 字段被允许供奖励使用，因此不能依赖任何隐藏在 info 中的成功标志。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 必须视为空字典（无字段可用）
- training_progress 当前 prompt 未说明其可用性，应默认不可用（除非后续明确允许）

禁止使用：
- original_reward（被严格屏蔽）
- 任何未在观测空间或 action 空间中声明的变量
- 任何 info 字段（因为是空字典，且未说明有隐藏字段）

## 7. 可用于奖励函数的信号
- position: 相对垫子的水平距离 x_position (obs[0])，高度 y_position (obs[1])
- velocity: 水平速度 x_velocity (obs[2])，垂直速度 y_velocity (obs[3])
- orientation: 身体角度 body_angle (obs[4])，角速度 angular_velocity (obs[5])
- contact: 左腿触碰 left_support_contact (obs[6])，右腿触碰 right_support_contact (obs[7])
- action/engine: 动作类型 (action)，可以推断燃料消耗（非零动作消耗）
- other: 从 next_obs 可计算速度变化、当前信息；终止可与接触模式结合判断着陆是否完全稳定。

## 8. 不确定或不可用的信号
- 成功/失败显式标志：不存在
- 目标垫的位置：相对位置已在观测中给出，不需要额外信息
- 燃料剩余量：未提供
- 时间/步数：未作为观测，不可用
- 风力/干扰：被屏蔽
- official reward：不可用
- info 字段：空

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_rigid_body_with_landing_legs
  actuator_type: thruster_based (main_engine + two_orientation_engines)
  contact_structure: two_legs (left, right)
primary_objectives:
  - reach_and_land_safely_on_target_pad (position + low velocity + upright orientation at touchdown)
secondary_objectives:
  - minimize_fuel_consumption
  - arrive_as_fast_as_possible (implicit, no time signal directly)
main_failure_risks:
  - crash_due_to_excessive_velocity_or_angle
  - miss_target_pad (drift outside viewport)
  - settle_on_side_or_upside_down (not qualified landing)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 鼓励接近目标垫（x和y方向）
  why_required: 没有位置奖励，任务无法收敛到正确位置。
  usable_signals: [x_position, y_position]（可直接用距离或分段奖励）
  risks: 若密度太大可能让agent停在附近但不降落；需要与速度/姿态约束配合。

- role_id: soft_landing
  purpose: 确保接近垫面时垂直速度小，双腿接触且姿态接近水平
  why_required: 单有位置奖励可能导致高速撞击或倾斜着陆，需要区分“成功停靠”的严格要求。
  usable_signals: [y_position, y_velocity, left_support_contact, right_support_contact, body_angle]
  risks: 过早引入可能导致探索不足（因为条件苛刻）；可仅在高度低于阈值时激活。

- role_id: fuel_efficiency
  purpose: 惩罚非零引擎动作以节省燃料
  why_required: 任务说明明确要求“as little engine thrust as possible”。
  usable_signals: [action]（非零动作即消耗）
  risks: 权重过高可能导致agent不敢用引擎，无法减速着陆；需与主目标平衡。

### 10.2 条件职责 conditional_roles
- role_id: orientation_stabilization
  condition_to_use: 当高度低于某个安全阈值或双腿接近地面时加强，防止侧翻
  usable_signals: [body_angle, angular_velocity]
  risks: 在远离目标时就限制姿态可能限制机动性；宜在终端阶段加强。

- role_id: speed_moderation (intermediate phase)
  condition_to_use: 当接近目标垫水平与垂直都在一定范围内时，鼓励减速但不强制完全停止，帮助平稳过渡。
  usable_signals: [x_velocity, y_velocity, x_position, y_position]
  risks: 过早限制速度可能导致上时间过长。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: survival_balance
  reason: 环境没有需要持续维持平衡的机制，只有最终着陆环节的稳定性要求，不应作为全局职责。
  forbidden_or_missing_signals: 无跌倒即死之类信号，终止由碰撞/出界/静止导致，生存职责不适用。

- role_id: sparse_exploration_bonus
  reason: 观测空间连续且提供清晰的距离和速度信息，不需要额外的探索职责；硬加到奖励会破坏稀疏奖励结构。
  forbidden_or_missing_signals: 无

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | None | dense_state_signal, bounded_signal (distance) | 可对水平距离和垂直距离分别建模，也可合成欧氏距离 |
| soft_landing | y_position, y_velocity, left_support_contact, right_support_contact, body_angle | ground_contact_normal_vector（未提供） | thresholded_penalty, quadratic_penalty, logical_success_bonus | 可在 height < eps 且双腿接触时给予大额成功奖励，否则对高速/倾斜施加惩罚 |
| fuel_efficiency | action | None | linear_penalty (action != 0) | 可直接用负奖励系数乘以非零动作的次数 |
| orientation_stabilization | body_angle, angular_velocity | None | quadratic_penalty | 通常使用 angle^2 + angular_vel^2 |
| speed_moderation | x_velocity, y_velocity (条件激活) | None | quadratic_penalty, dead_zone | 仅在接近目标区域时激活 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 永不启动主引擎（悬停或自由落体）| 平均episode长度短，y_velocity 持续负向且无减速 | 增加 goal_proximity 权重或提供高度下降惩罚 |
| 在垫外某处盲目降速并静止（假成功）| 终止时 x_position 或 y_position 不接近0，双腿不一定接触 | 强化 soft_landing 的精确位置条件，仅在垫上方给予成功奖励 |
| 撞击目标垫（高速或倾斜）| 终止于 crash，垂直速度大，身体角度大 | 调整 soft_landing 中的速度/角度惩罚阈值与系数 |
| 过度使用引擎（高燃料消耗）| episode 中非零动作比例极高，但成功着陆 | 提高 fuel_efficiency 惩罚，或改用功率积分 |
| 摇摆不定、迟迟不着陆 | 高度长时间在中等区域振荡，orientation 频繁校正 | 引入接近终端时的速度/角度强制收敛项，或者降低 orientation_stabilization 的过早激活 |
