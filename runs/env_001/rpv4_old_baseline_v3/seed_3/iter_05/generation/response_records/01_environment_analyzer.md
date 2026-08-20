# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 车辆轨迹优化问题。智能体从视口上方中心附近出发，需要尽可能快地飞到目标平台（中央垫板）并稳定停靠，同时尽量节省发动机推力。  
**主要目标**：准确到达目标位置（x≈0, y≈0）、软着陆（低线速度、低角速度、双支撑接触并稳定）。  
**次要目标**：燃料经济性（减少不必要的发动机使用），但仅在不牺牲着陆成功的情况下才考虑。  
**不应混淆的目标**：不是单纯的导航到达，必须在接触阶段满足姿态和速度约束，否则视为失败。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达一个指定的空间位置（目标平台），并附带姿态控制、速度抑制和接触安全性要求；该类问题的主干属于目标到达，附属约束可在奖励函数中作为条件角色加入。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0] (x_position): 相对于目标平台的水平坐标，可正可负 (reward_usable: true)
- obs[1] (y_position): 相对于垫板高度的垂直坐标 (reward_usable: true)
- obs[2] (x_velocity): 水平线速度 (reward_usable: true)
- obs[3] (y_velocity): 垂直线速度 (reward_usable: true)
- obs[4] (body_angle): 机体朝向角 (reward_usable: true)
- obs[5] (angular_velocity): 角速度 (reward_usable: true)
- obs[6] (left_support_contact): 左支撑腿接触标志 (0.0 或 1.0) (reward_usable: true)
- obs[7] (right_support_contact): 右支撑腿接触标志 (0.0 或 1.0) (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0 (no_engine): 不启动任何发动机，让机体自由飘移/受重力影响
- action 1 (left_orientation_engine): 启动左侧姿态发动机，产生旋转力矩（改变朝向）
- action 2 (main_engine): 启动主发动机，产生向机体上方的推力（主要进行高度/速度控制）
- action 3 (right_orientation_engine): 启动右侧姿态发动机，产生反向旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 当机体休眠时触发，可能暗示已稳定着地，但需要结合接触和速度进一步确认；也可能是其他原因导致不再活动。
- failure-like termination: crash_or_body_contact（机身与地面/其它物体异常碰撞）或 horizontal_position_outside_viewport（水平飞出视野）明确为失败。
- ambiguous termination: body_not_awake_or_settled 不能单独作为成功标志，必须检查观测中的双接触状态和速度。
- truncation: 无，或未声明，环境未提供步长限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false （info 未提供故障原因字段）
- allowed_info_fields: 无 （info = {}，禁止使用原始奖励或任何未声明的 info）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs （当前观测，所有维度均可）
- action （当前动作）
- next_obs （下一状态，用于检测是否达成的着陆条件或进度变化）
- 上述信号中已声明的字段

禁止使用：
- original_reward（已屏蔽）
- info（为空，不可用）
- training_progress（任务提示中未明确允许使用，不可用）
- 任何未在观察空间中显式声明的外部信号

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position)、obs[1] (y_position) — 相对目标位置，可构建距离、高度误差等
- velocity: obs[2] (x_velocity)、obs[3] (y_velocity) — 代表接近时的速度，可惩罚高速接触
- orientation: obs[4] (body_angle) — 应接近直立/目标姿态
- contact: obs[6] (left_support_contact)、obs[7] (right_support_contact) — 用于判断着陆完成
- action/engine: action 本身可判断是否使用了发动机（用于燃料惩罚）
- other: obs[5] (angular_velocity) 用于姿态稳定

## 8. 不确定或不可用的信号
- 无明确的“成功”标志位（如 terminal_info 中的 success），只能从 next_obs 的接触+速度+位置组合推断
- 无燃料或推力直接测量，只能通过是否执行动作 1/2/3 来近似能耗
- 无外部参考目标轨迹或距离理想路径的偏差信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_body_lander
  actuator_type: thruster (main + two orientation)
  contact_structure: two_leg_contact
primary_objectives:
  - reach_target_position (x→0, y→0)
  - stable_landing (low velocities, upright orientation, both legs contact)
secondary_objectives:
  - fuel_efficiency (minimize engine usage)
  - rapid_arrival (implicitly through discounting)
main_failure_risks:
  - crash_into_ground_or_pad_due_to_excessive_velocity
  - drift_out_of_viewport_horizontally
  - tumbling_or_uncontrolled_rotation_preventing_landing
  - never_activating_engines_and_missing_target
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: progress_to_target
  purpose: 指引智能体向目标平台移动，减小与目标的水平与垂直距离。
  why_required: 没有此职责，智能体没有方向性，难以找到着陆点。
  usable_signals: [x_position, y_position, 上一帧位置（通过 obs/next_obs 差分）]
  risks: 如果仅用距离惩罚，可能在目标附近徘徊而不会减速着陆。

- role_id: soft_landing
  purpose: 在接近目标区域时强制降低线速度、角速度，并最终确保双支撑稳定接触。
  why_required: 任何仅依赖位置奖励的策略都可能以高速撞击平台导致失败。
  usable_signals: [x_velocity, y_velocity, angular_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position]
  risks: 如果权重过高，可能导致智能体不敢下降；需配合进度奖励使用。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 最小化发动机使用次数或时长，从而节省燃料。
  condition_to_use: 在训练后期或当智能体已能稳定着陆时开启；或使用极小的负系数仅起辅助作用。初期不建议使用，以免抑制必要的推力探索。
  usable_signals: [action (0=无动作, 1/3=姿态, 2=主发动机)]
  risks: 若过早或过强，会造成智能体选择无动作而错过着陆窗口，甚至因重力坠落而失败。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_bonus_from_info
  reason: info 中没有 success 标志，无法使用。
  forbidden_or_missing_signals: [info.success]

- role_id: trajectory_smoothing (动作连续性)
  reason: 动作为离散 4 选 1，连续动作序列的平滑性难以定义，强行惩罚动作切换可能导致发动机频繁抖动（振荡），反而降低性能。
  forbidden_or_missing_signals: 无自然动作平滑度量，且可能与该离散系统的最佳控制模式冲突。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| progress_to_target | x_position, y_position (current & last) | 无 | dense_state_signal (欧氏距离), potential-based reward (距离的减少) | 避免仅用绝对距离，否则终点奖励过强 |
| soft_landing | x_velocity, y_velocity, angular_velocity, body_angle, left/right contacts, proximity (x,y) | 无 | bounded_signal (速度幅度), quadratic_penalty (超限部分), threshold event reward (双接触且速度<阈值) | 需根据接近程度动态开启，防止远处减速 |
| fuel_efficiency | action | 真实的推力积分 | sparse_action_penalty (每个非零动作给负值) | 极小的负值，或仅在着陆成功后回溯性奖励节能行为时使用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 什么都不做，自由落体坠毁 | 平均奖励低，动作直方图 action=0 占绝对主导 | 增加 progress_to_target 奖励权重，或加入很小的动作多样性奖励（仅用于早期探索） |
| 持续大推力水平高速飞出视口 | x_velocity 最终极大，x_position 超出阈值 | 增加水平速度惩罚，或增加位置边界接近时的警告奖励 |
| 到达目标上方但旋转失控无法稳定 | body_angle/angular_velocity 不停振荡，最终 crash | 增加姿态稳定奖励，特别是接近目标时加大角速度惩罚 |
| 频繁开关发动机，消耗高且轨迹不稳定 | 动作变化率高，双接触次数少 | 加入软着陆的强事件奖励，宁可稍慢也要降速，必要时降低燃料效率职责的影响 |
| 以单腿或倾斜方式撑着平台但不稳定 | 只有一条腿接触，或 body_angle 非零但 termination 发生 | 更严格定义双接触 + 小角度才算成功，使用事件奖励强调这些条件 |
