# Response Record

# 匿名环境理解卡片

## 1. 任务目标
任务目标是控制一个受随机初始力作用的二维飞行器，使其从视野顶部中央出发，尽可能快地飞抵并稳定停靠于中央目标平台上。核心目标是“到达并停靠”（reaching and settling），附属优化目标包括“尽可能少用引擎推力”和“保持姿态稳定、安全接触”。不应将其混淆为单纯的存活任务、无限制漫游任务或多目标博弈任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求“reach and settle at a central target pad”，核心目标是到达指定目标位置；附属目标为速度、能耗和姿态质量，符合导航到达的主任务范式。观测空间提供相对于目标的坐标，进一步证实其到达属性。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (推测连续量)
- obs[0]: x_position (水平方向相对于目标平台的坐标)，reward_usable: true
- obs[1]: y_position (垂直方向相对于平台高度的坐标)，reward_usable: true
- obs[2]: x_velocity (水平线速度)，reward_usable: true
- obs[3]: y_velocity (垂直线速度)，reward_usable: true
- obs[4]: body_angle (机体朝向角)，reward_usable: true
- obs[5]: angular_velocity (角速度)，reward_usable: true
- obs[6]: left_support_contact (左支撑腿接触标志, 0/1)，reward_usable: true
- obs[7]: right_support_contact (右支撑腿接触标志, 0/1)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力)，即不激活任何引擎
- action 1: left_orientation_engine (左姿态引擎)，产生顺时针或逆时针力矩中的一种；具体方向需在交互中推断，但用于调整朝向
- action 2: main_engine (主引擎)，沿机体纵轴提供推力，用于平移/减速/抵抗重力
- action 3: right_orientation_engine (右姿态引擎)，产生与左姿态引擎相反的力矩，用于反方向姿态修正

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止；任务期望通过“到达并稳定停靠”后 episode 结束，这很可能通过 **timeout/truncation** 或在目标区域达到低速度、双腿接触、小角度等条件后被环境内部判定为 settled 而终止。
- failure-like termination:
  - *crash_or_body_contact*: 机体部分（非支撑腿）碰撞地面/平台以外区域，或姿态严重偏离导致翻倒。
  - *horizontal_position_outside_viewport*: 机体飞出水平边界，视为严重失控。
  - *body_not_awake_or_settled*: 可能是检测到速度/加速度极小但未达成着陆条件，或进入睡眠状态的超时机制。
- ambiguous termination: 支撑腿接触目标平台但未满足所有稳定条件，被 terminated 可能属于部分成功/硬着陆，不能直接视为完美成功。
- truncation: 任务可能包含 episode 长度上限，届时会直接截断。该截断不携带成功/失败固有语义。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 根据 source，step 返回空字典 `{}` ，因此 `info` 无任何可用字段。
- forbidden_or_uncertain_info_fields: info字典为空，无字段可用；不得依赖任何隐式 info 键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 包含步骤执行前的状态（8维）
- action: 执行的动作（0~3）
- next_obs: 执行动作后的状态（8维）
- info: 必须为空字典，禁止访问任何字段
- training_progress: 本次 prompt 并未明确允许或禁止；保守做法是仅在绝对必要时使用，且不得作为唯一主信号

禁止使用：
- original_reward
- official_reward
- 任何未在 info 中声明的字段
- 任何未在 obs 空间中声明的隐藏状态

## 7. 可用于奖励函数的信号
- position: `next_obs[0:2]` 表示相对于目标平台的位置。可计算当前距离、距离变化量。
- velocity: `next_obs[2:4]` 线速度。可用于接近速度、稳定着陆时趋零、水平漂移控制。
- orientation: `next_obs[4]` 机体角度。可用于姿态维护、着陆时接近水平的奖励/惩罚。
- contact:
  - `next_obs[6]` 左支撑腿接触
  - `next_obs[7]` 右支撑腿接触
  - derived_possible: 双腿同时接触（legs_contact = left & right）是成功着陆的关键条件，可直接从观测构造。
- action/engine: `action` 可以用于对引擎使用施加惩罚。
- other:
  - angular_velocity `next_obs[5]` 可用于控制姿态抖动的阻尼惩罚。
  - derived_possible: settled 成功事件可间接推断：如果 episode 未因 crash/越界终止而截断，且最后几步保持双腿接触、低速度、小角度，则很可能为成功着陆。可在最终奖励中使用 sparse terminal success bonus，但必须标注为 derived_possible，且需在策略中小心处理以避免误判。

## 8. 不确定或不可用的信号
- 精确的成功标记 (info-based): 不存在。
- 燃料/推力能量消耗绝对值: 未直接提供推力大小，仅知动作选择，因此只能对“使用引擎”这一动作本身进行惩罚，无法得知实际推力大小。
- 地面/平台的法向量或确切碰撞位置: 不可用。
- 风向/随机力强度: 未暴露，无法直接补偿。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (vehicle/lander)
  actuator_type: discrete thrusters (1 main + 2 orientation)
  contact_structure: two support legs (left/right contacts)
primary_objectives:
  - reach target pad (minimize distance to goal)
  - achieve soft landing (low velocity, near-zero angle, both legs in contact)
secondary_objectives:
  - minimize engine usage (actions 1,2,3 penalized)
  - minimize time-to-land (implicit in fast approach)
main_failure_risks:
  - crashing body parts other than legs
  - exiting horizontal bounds
  - excessive angle / angular velocity leading to instability
  - hard landing with high linear velocity
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: delta_distance_to_goal
  purpose: 推动 agent 向目标平台移动，是“更快到达”的核心驱动力。
  why_required: agent 初始位于顶部中央，必须克服随机初始力抵达中央。若无此信号，agent 可能悬停或随意飘浮。
  usable_signals: [next_obs[0], next_obs[1], obs[0], obs[1]]
  risks: 仅依赖 proximity (当前距离) 会导致 agent 停在半空不好不坏的平衡点而不着陆；因此必须使用 delta 形式 (distance_curr - distance_next) 或 improvement 来消除悬停陷阱。

- role_id: soft_landing_terminal_bonus
  purpose: 在 episode 结束时给予成功软着陆的明确奖励，将“到达”与“安全停靠”绑定。
  why_required: delta_distance 仅保证接近，不能保证减速、姿态对齐和双腿接触。必须用 terminal bonus 将优化引向最终的稳定状态。
  usable_signals: [next_obs[6], next_obs[7], next_obs[2:4], next_obs[4]]; derived_possible 判定 terminal 是否为成功 (非 crash/越界)。
  risks: bonus 过大会扭曲飞行阶段的决策；应设为适度值。判定成功需组合多个条件，误判会导致奖励污染。

### 10.2 条件职责 conditional_roles
- role_id: health_constraint
  condition_to_use: 当 agent 接近危险状态时激活，防止在接近目标时 crash。不应全程施加，避免初期探索受挫。
  usable_signals: [next_obs[4] 大角度, next_obs[0] 出界风险, derived_possible crash 接触 (可能来自角度突变或速度骤降)]
  risks: 过强的姿态/出界惩罚会阻碍早期探索，导致 agent 不敢移动或不敢调整姿态。应采用 hinge loss 仅在超出安全阈值时惩罚。

- role_id: engine_usage_penalty
  condition_to_use: 仅在任务明确要求“尽可能少用引擎”时加入，且通常作为弱正则项。不建议在到达目标前给予过大权重，否则 agent 可能选择不点火而飘离目标。
  usable_signals: [action == 1,2,3]
  risks: 过度惩罚会与“尽快到达”冲突，使 agent 行动迟缓。权重应远小于 delta_distance 奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: constant_survival_bonus
  reason: 生存不是主任务；给予持续存活奖励会制造悬停收割陷阱，使 agent 满足于不坠毁而不着急着陆，与“尽快”冲突。
  forbidden_or_missing_signals: 无独立存活信号。

- role_id: angular_velocity_smoothness
  reason: 虽然姿态平稳是好事，但单独的成分在多数抵达目标环境中会与“快速接近”构成不必要的折衷。若观测发现姿态振荡严重，再作为微弱阻尼添加。初始版本禁用。
  forbidden_or_missing_signals: 无。

## 11. role_to_signal_mapping
| role_id                     | usable signals                                              | missing signals | candidate formula operators                                        | notes                                                                                              |
| --------------------------- | ----------------------------------------------------------- | --------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| delta_distance_to_goal      | obs[0:1], next_obs[0:1]                                    | —               | delta(distance), bounded_signal, improvement                       | 绝对距离作为 fallback 信号，但主信号必须是距离减量或类似进步度量。                   |
| soft_landing_terminal_bonus | next_obs[6], next_obs[7], next_obs[2:4], next_obs[4]       | —               | sparse terminal indicator, success_condition                       | 需 derived_possible 推断成功终止。组合双腿接触、低速度、小角度判定。                 |
| health_constraint           | next_obs[4], next_obs[0], derived_possible crash 推测      | —               | hinge_penalty, bounded_penalty                                     | 只在超出阈值时启用；不要用 quadratic penalty 在安全区内也惩罚。                       |
| engine_usage_penalty        | action (1,2,3)                                              | —               | constant_penalty                                                   | 权重必须很小，防止抑制探索。                                                                   |
| survival_bonus (avoid)      | —                                                           | —               | —                                                                  | 与“尽快”冲突。                                                                                 |
| angular_smoothness (avoid)  | next_obs[5]                                                 | —               | —                                                                  | 避免过早引入多目标权衡；若抖动严重再作为弱阻尼加入。                                          |

## 12. 初始训练后应观察的 failure modes
| failure_mode                         | evidence_to_check                                                              | possible_intervention                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 悬停不降落                            | episode 时间耗尽；agent 停留在目标上方但距离不再下降                           | 检查 delta_distance 是否被正确实现为 improvement；禁用生存奖励或降低生存相关权重。             |
| 快速冲向地面但 crash                  | 高速垂直接地、双腿未接触、episode 以 crash 终止                                 | 提高 soft_landing_terminal_bonus 的条件严格度；引入 hinge penalty 限制最大允许接地速度。       |
| 只用一个支撑腿接触，姿态歪斜          | 一侧接触为 1 另一侧为 0，body_angle 非零                                      | 在 terminal bonus 中要求双腿同时接触；或者在接近目标时添加姿态对齐的微弱辅助奖励。              |
| 大量使用引擎但进展缓慢                | 频繁使用 main_engine 但 Δdistance 小；燃料消耗高但前进少                       | 确认 delta_distance 信号是否足够强；考虑增加成功着陆 bonus 的吸引力，超过引擎惩罚。             |
| 胆小不点火，被风吹出视野              | 动作始终为 0，next_obs[0] 快速增大 (出界)                                      | 暂时降低 engine_penalty；确认 delta_distance 能提供足够梯度鼓励向目标移动；增加出界 penalty。    |
| 姿态失控旋转                          | angular_velocity 或 body_angle 一直增大至 crash                                | 先不修改奖励，检查策略探索是否得到足够的姿态动作；若持续，再引入 angular_velocity hinge penalty。 |
