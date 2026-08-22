# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 飞行器轨迹优化问题。飞行器从靠近视口顶部中心的位置出发，并带有初始随机力干扰。核心目标是**尽快到达并稳定停在中心目标平台上**（着陆），同时尽可能少用引擎推力。Agent 需要学习接近目标、减速、保持姿态稳定，并与平台软接触（安全着陆）。次要目标包括时间最短和燃料消耗最少，但它们是附属指标，不影响任务成功定义。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务的核心是到达指定的中心目标平台，附属目标（快速、省燃料）是性能优化，并不构成另一核心目标。没有存活或平衡作为独立目标，也没有操纵物体或多目标冲突。完全匹配“基于目标的到达任务”。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact
原因：飞行器不仅要到达目标区域，还必须减速、保持安全姿态，并实现稳定接触（着陆）。动作离散，包含姿态控制，符合“接近目标并低速、稳定接触/停靠”的特征。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (假定)
- obs[0]: x_position —— 飞行器相对目标平台中心的水平偏移（单位待定，通常米级），reward_usable: true
- obs[1]: y_position —— 飞行器相对平台高度的垂直偏移，reward_usable: true
- obs[2]: x_velocity —— 水平线速度，reward_usable: true
- obs[3]: y_velocity —— 垂直线速度，reward_usable: true
- obs[4]: body_angle —— 飞行器倾斜角（弧度），reward_usable: true
- obs[5]: angular_velocity —— 角速度，reward_usable: true
- obs[6]: left_support_contact —— 左支撑点接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact —— 右支撑点接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine —— 不点火（无推力）
- action 1: left_orientation_engine —— 点火产生左转力矩（姿态引擎）
- action 2: main_engine —— 主引擎点火（产生向上的推力，可能也有侧向推力的影响，需根据物理推断）
- action 3: right_orientation_engine —— 点火产生右转力矩（姿态引擎）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 当飞行器静止且稳定（身体进入休眠状态或接触稳定）时终止，此时极有可能成功着陆。但由于环境 info 为空，无法直接确认。
- failure-like termination: crash_or_body_contact —— 通常意味着飞行器与地面或平台以外区域发生剧烈撞击（例如腿折断、速度过大撞击导致销毁）。观察中 crash 不能直接读出，但可通过速度和角度突变或接触信号组合推断。
- failure-like termination: horizontal_position_outside_viewport —— 飞行器飞出水平边界，必然失败。
- ambiguous termination: 某些身体接触终止可能同时满足 crash 和 settled 条件，但源代码中用 `or` 连接，且优先判定终止；失败/成功的区分只能通过观察状态结合终止方式推断。
- truncation: 未提及最高步数截断（假设有 episode 长度限制，但不暴露在 info 中）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，不能从中读取任何字段）
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用（包括 success, failure, crash, 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步观察（8维数组）
- action: 当前步执行的动作（整数0-3）
- next_obs: 下一步观察（8维数组）
- info: 空字典（可读取但无可用字段）
- training_progress: 除非 prompt 明确要求，否则不使用

禁止使用：
- original_reward（官方奖励）
- 任何不在 info 中的额外字段
- 真实环境名称或内部状态

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])，相对于目标平台的水平、垂直偏移
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4])，可用于姿态惩罚；angular_velocity (obs[5]) 可用于姿态平滑性
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])：着陆接触信号，可用于判断是否已接触并稳定
- action/engine: 动作本身可推断推力使用（如 main_engine 消耗燃料，orientation engines 可能也消耗微量），可用于燃料效率奖励
- derived_possible: 
  - 成功着陆推断：当 episode 终止（terminated）且 body_angle 接近 0、速度极小、左右接触均为 1、位置接近 (0,0) 时，可视为成功着陆（需要从 next_obs 或终止时观察判断，但 reward 函数在终止步仍会被调用，此时 next_obs 为最终状态，可使用）
  - 坠毁推断：当终止且速度或角度突变过大、接触点不完整或位置明显偏离目标区域
  - 出界推断：当 |x_position| 超出视口范围（可从位置推断，通常是 [−1,+1] 级别，但确切范围未知，需保守）

## 8. 不确定或不可用的信号
- 官方奖励 original_reward：被屏蔽，不可用
- 精确的成功/失败 flag：info 为空，无任何标志
- 仿真时间或 episode 步数：未在观察或 info 中提供，不可用
- 燃料剩余量：未提供，只能从动作间接估计消耗
- 风速/随机力：不可见

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body with two landing legs/supports
  actuator_type: main engine (thrust) + two orientation thrusters (torque)
  contact_structure: two-point support (left/right), contact flags indicate ground touch
primary_objectives:
  - reach target pad center (minimize distance to (0,0))
secondary_objectives:
  - minimize fuel usage (limit engine firings)
  - minimize time to landing (encourage fast approach)
main_failure_risks:
  - crashing at high speed or wrong angle
  - drifting out of horizontal viewport
  - tipping over (large body_angle), preventing stable settled state
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: progress_to_target_by_delta_distance
  purpose: 驱动飞行器持续向目标平台移动，同时避免悬停收割（仅在距离缩小时获得正奖励）
  why_required: 任务核心是到达目标；使用 delta(distance) 防止 agent 静止在远处仍获得正向奖励（proximity 陷阱）
  usable_signals: [x_position, y_position] （可从 obs 和 next_obs 计算欧氏距离变化）
  risks: 如果移动但效率低仍获得微小正奖励，可能需要补充时间惩罚或稀疏事件奖励以加速学习

- role_id: soft_landing_terminal_bonus (derived possible)
  purpose: 在成功着陆时提供强正信号，加速学习终端行为
  why_required: 稀疏的终极目标需要终端奖励来明确成功条件，尤其在 delta 距离信号不够强时
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact] 以及 episode 终止标志（可直接从环境 terminator 在 reward 计算时传入，无需 info）
  risks: 需要谨慎定义成功条件，避免因误判给予虚假成功奖励

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  condition_to_use: 始终启用（任务描述明确要求尽量少用推力）
  usable_signals: [action] —— 当 action == main_engine (2) 时，消耗燃料；转向引擎（1 and 3）也可能消耗少量，可选
  risks: 若惩罚过重，可能导致 agent 不敢使用主引擎，始终飘浮，无法到达目标；应保持 delta 距离主奖励足够强，且仅在接近目标时适当加重节能权重（可通过距离门控或分阶段调整）

- role_id: attitude_stability_bonus
  condition_to_use: 在飞行器接近目标且需要减速着陆时尤为重要（可随距离衰减启用）
  usable_signals: [body_angle, angular_velocity]
  risks: 在远距离转向阶段不应该过分惩罚，否则妨碍调整方向；建议在距离小于某阈值时生效，或者只惩罚超出安全范围的倾斜角（hinge 形式）

### 10.3 慎用/禁用职责 avoid_roles
- role_id: proximity_reward (state-based distance)
  reason: 如果给予当前距离的单调递减函数（如 -distance），agent 可以停在远处持续获得正奖励，而不继续前进。这与“尽快到达”冲突，必须避免。
  forbidden_or_missing_signals: 不缺失信号，但其数学形态容易导致行为退化。

- role_id: explicit_time_penalty
  reason: 没有直接的步数计数可用（info 为空），无法全局计时。可采用隐式方式通过逐步微小的负奖励来鼓励快，但这会影响探索，且与节约燃料目标可能冲突。
  forbidden_or_missing_signals: 缺少全局步数或时间信号；如果使用 constant per-step penalty，可能导致 agent 过早停止移动，不推荐单独使用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| progress_to_target_by_delta_distance | x_position, y_position (obs & next_obs) | none | delta_distance = distance(current) - distance(next) (positive when reducing) can be bounded to avoid large jumps | 避免悬停的关键；如果 delta 为负（远离），可给予小惩罚 |
| soft_landing_terminal_bonus | x_position, y_position, x_velocity, y_velocity, body_angle, left_contact, right_contact, termination flag | explicit success flag | 条件判断: terminated & distance<ε & speed<ε & |angle|<ε & L&R contact → bonus | 终端奖励必须在环境step返回terminated=True且该步reward函数被调用时给出；若环境在终止步不调用reward，则无法实现，需另行方式 |
| fuel_efficiency_penalty | action | fuel measurement | penalty = coefficient * (1 if main_engine else 0) + optionally smaller for orientation engines | 可以乘以一个距离相关 gating factor，在远距离时权重低 |
| attitude_stability_bonus | body_angle, angular_velocity | none | hinge_penalty: max(0, |angle| - safe_threshold) and |angular_vel| * weight | 用于着陆阶段近端；可用距离门控 `if distance < threshold` |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停而不降落 | 飞行器悬在目标上方远方，x,y 位置长时间不变化，但 reward 未归零（如果用了 proximity） | 确保主体奖励为 delta distance，删除任何 proximity 信号；可增加 fuel 惩罚或加 terminal 奖励诱导 |
| 从不点火主引擎 | 动作频繁选择 0,1,3，速度下降很少，timeout 终止 | 减弱 fuel 惩罚，或仅在距离较远时减少燃料惩罚权重；确保 delta 距离奖励足够大以覆盖燃料成本 |
| 高速撞击平台失败 | episode 因 crash 终止，接触点可能非双支撑，角度和速度大 | 增加与距离相关的速度上限惩罚（速度超过阈值给予负奖励），或采用速度门控的终端奖励安全条件 |
| 翻倒（大角度终止） | body_angle 在终止时数值大，接触点仅单侧 | 加入角度 hinge 惩罚，且在接近目标时加重；可在 early training 后加入 stability 条件 |
| 飞出水平边界 | x_position 大幅超出安全区域，terminated out-of-viewport | 增加位置出界前的大惩罚（如 |x|>1.0 时给予强负奖励），或通过 delta 距离惩罚使其远离边界 |
| 着陆但不稳定（bouncing） | 着陆后速度仍有较大跳跃，可能 settled 条件不满足，终止于 crash | 观察速度变化，可奖励低速度、低角速度，在近端强化 soft landing 条件 |
