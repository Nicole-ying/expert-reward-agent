# Response Record

# 匿名环境理解卡片

## 1. 任务目标
让一个 2D 飞行器/着陆器从初始随机扰动开始，尽快到达并稳定在中心目标平台上，同时尽量少用引擎推力。核心目标是实现安全、低速的接触与稳定停泊；次要目标是减少燃料消耗和缩短耗时。不应将“稳定停泊”与单纯的“到达目标位置”混淆：最终必须让左右支撑腿都接触地面（左/右接触标志为真），速度与角速度足够小，且身体角度接近水平，才算成功。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定目标平台并稳定停泊，具有明确的终点状态（位置、接触），虽附带节能与时间要求，但这些是明确的附属优化项，不足以构成多目标冲突，因此归入“目标到达”任务族。

动力学更细子类：goal_approach_and_soft_contact，因为任务强调低速接近、稳定接触与角度控制，具有明显的接近目标并实现软接触的特征。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 连续浮点（float32/float64），后两维为 0/1 但编码为连续值
- obs[0]: name=`x_position`, meaning=水平坐标相对于目标平台中心的偏移，reward_usable: true
- obs[1]: name=`y_position`, meaning=垂直坐标相对于平台高度的偏移，reward_usable: true
- obs[2]: name=`x_velocity`, meaning=水平线速度，reward_usable: true
- obs[3]: name=`y_velocity`, meaning=垂直线速度（向上为正或向下为正），reward_usable: true
- obs[4]: name=`body_angle`, meaning=本体方向角（弧度或度），reward_usable: true
- obs[5]: name=`angular_velocity`, meaning=角速度，reward_usable: true
- obs[6]: name=`left_support_contact`, meaning=左支撑腿接触地面时为 1.0，否则 0.0，reward_usable: true
- obs[7]: name=`right_support_contact`, meaning=右支撑腿接触地面时为 1.0，否则 0.0，reward_usable: true

所有维度在奖励函数中均可使用，因为它们描述了任务目标直接相关的状态。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine —— 不开启任何引擎，无推力消耗
- action 1: left_orientation_engine —— 开启左侧定向引擎，可能产生逆时针旋转和/或横向推力
- action 2: main_engine —— 开启主引擎，通常产生向上的推力（用于减速/悬停）
- action 3: right_orientation_engine —— 开启右侧定向引擎，可能产生顺时针旋转和/或横向推力

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 当物体处于休眠状态或已稳定在接触表面时触发。这通常发生在着陆并满足静态稳定条件后，应被理解为成功。但需注意：该终止也可能在远离目标的位置意外休眠时发生，此时并非真正成功。
- failure-like termination: `horizontal_position_outside_viewport` —— 水平位置超出视口范围，明确失败。
- ambiguous termination: `crash_or_body_contact` —— 任何身体与环境几何体发生严重碰撞或任意本体部分接触地面。若为高速碰撞、翻滚或仅部分接触，则为失败；但若为期望的左右支撑腿同时接触平台，则可能同时被 `body_not_awake_or_settled` 捕获。单独该终止可能意味着硬着陆或非稳定接触，因此单独不可直接判定为成功。
- truncation: 源代码中 `truncated` 参数固定为 `False`，环境不会因为步数上限截断，但训练中通常会有外部时间限制，在此环境卡片中忽略。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无任何标志）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用，因为环境不提供任何额外信息。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察（所有维度）
- action: 当前动作
- next_obs: 下一时刻观察（所有维度）
- info: 空字典，不允许依赖任何键
- training_progress: 本任务描述未提及启用该参数，故应假定禁止使用

禁止使用：
- original_reward（已掩码的官方奖励）
- official_reward 或其任何变体
- 未在允许列表中出现的任何附加数据

## 7. 可用于奖励函数的信号
- position: next_obs[0] (x_position), next_obs[1] (y_position) —— 用于判断是否靠近目标并处于平台高度
- velocity: next_obs[2] (x_velocity), next_obs[3] (y_velocity) —— 用于鼓励低速接触
- orientation: next_obs[4] (body_angle), next_obs[5] (angular_velocity) —— 用于保持水平姿态与稳定
- contact: next_obs[6] (left_support_contact), next_obs[7] (right_support_contact) —— 用于确认两腿着地
- action/engine: action 值 —— 可用来惩罚使用引擎（非零动作），以体现节能
- other: 无

## 8. 不确定或不可用的信号
- 是否已成功：无明确 success flag；需要根据终止时刻的 next_obs 接触状态、速度、位置等综合推断。
- 燃料量或燃料消耗率：不可用，只能通过动作选择间接表达节能偏好。
- 生存时间或步数：不可用（truncation 不存在，且未提供 episode 长度）。
- 任何与地形或平台形状相关的先验信息：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_lander
  actuator_type: discrete_thrusters (main + orientation engines)
  contact_structure: two_leg_contact (left/right support)
primary_objectives:
  - reach target pad with near-zero horizontal and vertical offset
  - establish stable contact with both legs on the pad
  - maintain near-horizontal orientation
secondary_objectives:
  - minimize engine usage (fuel conservation)
  - minimize time-to-contact
main_failure_risks:
  - overshooting or undershooting the pad due to excessive thrust
  - landing too fast, causing crash or bounce
  - tipping over (one leg contact or large body angle)
  - drifting out of viewport laterally
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_approach_progress
  purpose: 引导 agent 从任意初始状态向目标平台移动，缩小 x,y 距离
  why_required: 这是任务存在的根本动机，没有该职责 agent 不会学习接近目标
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 仅靠距离奖励可能导致高速接近、碰撞反弹或无视角度/接触质量

- role_id: soft_contact_stability
  purpose: 鼓励在目标区域双足着地且速度、角速度足够小，实现稳定停泊
  why_required: 任务明确要求“settle at the pad”，仅位置接近是不够的，必须实际稳定接触
  usable_signals: [next_obs[2], next_obs[3], next_obs[4], next_obs[5], next_obs[6], next_obs[7]]
  risks: 如果设计不当，可能奖励假接触（一腿触地、另一腿悬空）或保留小速度导致滑动

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency_penalty
  condition_to_use: 在 agent 已能较可靠地完成着陆后（训练后期）引入，以避免早期阻碍探索
  usable_signals: [action] （非零动作即为开启引擎，简单惩罚）
  risks: 过早施加可能导致 agent 拒绝采取任何推力，完全不探索接近目标

- role_id: orientation_keeping
  condition_to_use: 可始终使用，但其强度需要调低，以免与接近目标冲突
  usable_signals: [next_obs[4], next_obs[5]]
  risks: 过度强调水平可能造成 agent 拒绝必要的姿态调整

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_penalty
  reason: 环境未提供时间或步数的显式信号，info 为空且无 truncation，无法直接惩罚耗时。虽可通过恒定负奖励近似，但缺乏依据且容易溢出
  forbidden_or_missing_signals: [episode_length, elapsed_time, truncation_flag]

- role_id: direct_success_bonus
  reason: 无可用的 explicit success flag，终止条件混叠，判别逻辑复杂且危险（可能奖励 crash 式的终态），必须避免。
  forbidden_or_missing_signals: [info["success"]]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_approach_progress | next_obs[0], next_obs[1] | — | bounded_signal (e.g., exp(-||pos||)) | 使用平滑衰减，避免在远离时梯度消失 |
| soft_contact_stability | next_obs[2:8] | explicit_success_flag | logical_anded_indicators, quadratic_penalty on velocities, bounded_reward for dual contact | 需组合 contact 和速度阈值，注意避免惩罚探索阶段的短暂接触 |
| energy_efficiency_penalty | action | fuel_consumption | discrete_action_cost (1 if action!=0 else 0) | 训练初期最好禁用或设极低权重 |
| orientation_keeping | next_obs[4], next_obs[5] | — | bounded_signal on abs(angle) + squared_penalty on angular_velocity | 可用于始终辅助稳定，但权重应弱于位置项 |
| time_penalty | — | episode_length, truncation | — | 排除，无可用信号 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| agent 悬停于远处不做接近 | 平均 x,y 距离不随时间下降 | 提高 goal_approach_progress 权重或衰减尺度，确保远距离仍有梯度 |
| 高速撞击平台后弹飞或 crash | 终止前 y_velocity 极负，出现 crash 终止 | 强化 soft_contact_stability 对速度的惩罚，可引入接近地面时的速度上限逻辑 |
| 单腿着地、翻倒 | left/right contact 不同时为 1，body_angle 较大 | 增加双足同时接触的奖励信号，强化 orientation_keeping |
| 引擎持续全开耗能极高 | 动作序列中零动作占比极低 | 适时激活 energy_efficiency_penalty，并渐增其系数 |
| 在水平方向上漂移出界 | x_position 频繁超过视口限制导致终止 | 在 goal_approach_progress 中加入横向边界软约束，或惩罚大横向偏离 |
