# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主任务是控制一个二维刚体（类飞行器/着陆器）从顶部中央区域出发，在受到随机初始扰动力的情况下，尽快移动到画面中央的目标着陆区，并以低速、稳定姿态完成安全接触（“软着陆”）。次要目标是尽量减少引擎使用次数/推力消耗，追求燃料效率。不应将“存活/不坠毁”本身当作主目标——存活只是安全着陆的必要条件，而不是与到达目标并列的独立主目标；同时也不应把“悬停不动”误当作成功，因为任务要求最终稳定接触目标垫而非无限期悬空。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 观测空间明确给出相对目标的位置（x, y 相对 pad），终止条件包含接触/越界/静止判定，动作是离散的姿态与主引擎控制，整体结构是“接近并稳定停靠到一个固定目标区域”，符合到达目标位置为核心、姿态与能耗为附属要求的定义，而非持续前进型运动（无地形推进）、无操控物体行为、无显式多目标冲突、也非纯稀疏探索。

dynamics_subtype: goal_approach_and_soft_contact
reason: 观测中存在相对目标坐标、线速度、角度、角速度以及两个接触点标志，动作包含主引擎和左右姿态引擎，典型对应“接近目标并低速稳定接触”这一子形态，而不是双足/多足步态或抓取阶段任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 未在源码中显式声明，推测为 float32（未确认，需以实际环境返回类型为准）
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true
- obs[1]: y_position，相对目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体姿态角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右侧接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不启动任何引擎
- action 1: left_orientation_engine，启动左侧姿态引擎（产生反向力矩/侧向修正）
- action 2: main_engine，启动主引擎（主要提供向上/减速推力）
- action 3: right_orientation_engine，启动右侧姿态引擎（与 action 1 相对的修正方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功标志，但 `body_not_awake_or_settled`（机体进入静止/休眠状态）在其发生时若伴随 obs[6]、obs[7] 均为 1 且位置接近目标（obs[0]、obs[1] 接近 0）、速度接近 0，可视为“成功着陆”的间接证据；仅当满足这些位置+速度+接触组合条件时才可推断为成功，不能仅凭该终止标志本身判定成功。
- failure-like termination: `crash_or_body_contact`（发生坠毁/异常碰撞）通常对应失败；若发生该终止但同时接触点状态异常（例如姿态角过大、速度过大）则更确定为失败。
- ambiguous termination: `horizontal_position_outside_viewport`（水平方向越界）本身既可能是失败（飞出边界），但也可能与正常物理仿真边界裁剪有关，需结合速度/位置综合判断，不能单独作为成功或失败的确定信号。
- truncation: 源码中固定返回 `False`，即当前 masked step 未展示基于时间步数的截断机制；若存在最大步数限制，需以外部 wrapper 确认，此处不能假设。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（`info={}`，源码未声明任何字段）
- forbidden_or_uncertain_info_fields: `info["success"]`、`info["failure"]`、`info["termination_reason"]` 及任何未在 step 源码中声明的字段均不可假设存在，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（8 维状态，全部字段均可用）
- action（离散动作 0-3）
- next_obs（8 维状态，全部字段均可用）
- info 中明确允许的字段：无（当前环境 info 为空字典，没有可用字段）
- training_progress：本任务无明确课程学习信号需求，除非后续明确要求渐进式塑形，否则不建议使用

禁止使用：
- original_reward
- official_reward
- 任何未在 step 源码中声明的 info 字段（如 success、failure、termination_reason 等）
- 未声明的 obs 切片（仅可使用 obs[0]~obs[7] 这 8 个已定义维度）

## 7. 可用于奖励函数的信号
- position: obs[0]（相对目标水平位置）、obs[1]（相对目标垫垂直位置）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（姿态角）、obs[5]（角速度）
- contact: obs[6]（左接触）、obs[7]（右接触）
- action/engine: action 本身（0=无动作，1/3=姿态引擎，2=主引擎），可用于估算引擎使用频率/能耗代理
- other: terminated 标志（由环境返回，可用于判断阶段结束，但需结合 obs 状态推断具体原因，不能直接读取“原因文本”）

## 8. 不确定或不可用的信号
- info 字典目前为空，任何形式的 info 字段引用都不可用
- `crash_or_body_contact` 与 `horizontal_position_outside_viewport` 的具体触发原因不会以布尔量形式单独暴露给 reward 函数，只能通过 obs 状态（速度、角度、位置、接触）间接推断，存在一定不确定性
- `body_not_awake_or_settled` 不能保证等价于“成功”，仅在特定 obs 组合下才可近似成立
- dtype 未在提供材料中确认，若下游需要精确类型处理应以实际返回值核实
- 是否存在最大步数截断机制未知，不能假设 truncated 会被置为 True

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_rigid_body_2d_vehicle
  actuator_type: discrete_thruster_set  # main engine + two orientation engines
  contact_structure: dual_point_contact  # left/right support contact flags
primary_objectives:
  - 从初始扰动状态导航至目标垫中心（水平+垂直位置误差趋近于0）
  - 在接触目标时保持低速（水平/垂直速度趋近于0）
  - 在接触时保持稳定姿态（角度与角速度趋近于0）
secondary_objectives:
  - 减少引擎使用频次/总推力消耗（燃料效率）
  - 尽快完成任务（时间效率）
main_failure_risks:
  - 坠毁或异常碰撞（crash_or_body_contact）
  - 水平方向飞出可视区域边界
  - 高速或大角度接触导致的不稳定“硬着陆”
  - 长时间悬停不着陆（拖延但未坠毁）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_distance_shaping
  purpose: 引导智能体从初始位置逐步靠近目标垫中心
  why_required: 这是任务的核心目标，缺少该职责智能体无法学会导航
  usable_signals: [obs.x_position, obs.y_position, next_obs.x_position, next_obs.y_position]
  risks: [仅用距离塑形可能导致智能体学会缓慢接近而不着陆；需与接触/终止信号配合]

- role_id: velocity_damping_near_target
  purpose: 鼓励智能体在接近目标时降低线速度，为安全接触做准备
  why_required: 若不约束速度，智能体可能高速冲向目标造成坠毁，无法实现“软着陆”
  usable_signals: [obs.x_velocity, obs.y_velocity, next_obs.x_velocity, next_obs.y_velocity, obs.x_position, obs.y_position]
  risks: [若权重过大可能导致智能体在远离目标时也不敢加速，拖慢整体进度]

- role_id: orientation_stability
  purpose: 保持机体姿态角与角速度接近稳定值，避免翻转
  why_required: 姿态失控通常直接导致坠毁类终止，是安全接触的必要条件
  usable_signals: [obs.body_angle, obs.angular_velocity, next_obs.body_angle, next_obs.angular_velocity]
  risks: [过度惩罚姿态可能限制必要的姿态调整动作，需与接近目标阶段解耦]

- role_id: terminal_outcome_signal
  purpose: 在回合结束时，根据是否在合理位置+速度+接触状态下终止，给予明确的终止性奖励/惩罚
  why_required: 稀疏但关键的终局反馈，帮助智能体区分“成功着陆”与“坠毁/越界”两类结局
  usable_signals: [terminated 标志, next_obs 全部字段（用于组合判断是否满足成功条件）]
  risks: [由于没有显式 success/failure 标志，需要自行用位置+速度+接触组合近似判断，存在误判风险，应做保守阈值设计]

### 10.2 条件职责 conditional_roles
- role_id: contact_bonus
  condition_to_use: 仅当 obs.left_support_contact 或 obs.right_support_contact 从0变为1，且此时位置/速度/姿态均在合理范围内时才触发，用于强化“正确时机的接触”而非鼓励提前触地
  usable_signals: [obs.left_support_contact, obs.right_support_contact, next_obs.left_support_contact, next_obs.right_support_contact, position, velocity]
  risks: [若不加位置/速度约束，可能被智能体利用为“任意时刻触地即可获奖”的漏洞]

- role_id: engine_usage_penalty
  condition_to_use: 仅在主任务（接近目标、稳定姿态）已能有效学习之后，作为次要塑形项加入，用于鼓励节省燃料
  usable_signals: [action]
  risks: [若过早或过重引入，可能导致智能体不敢使用引擎调整轨迹，从而学不会到达目标]

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_failure_bonus
  reason: 环境未提供 info["success"]/info["failure"] 等显式标志，无法直接读取官方判定结果，只能通过 obs 状态和 terminated 标志间接近似，若直接假设该字段存在会违反接口约束
  forbidden_or_missing_signals: [info.success, info.failure, info.termination_reason]

- role_id: time_step_penalty_from_truncation
  reason: 当前 masked step 源码中 truncated 恒为 False，没有证据表明存在基于步数的截断机制，不应基于假设的最大步数设计惩罚
  forbidden_or_missing_signals: [info.max_steps, info.elapsed_steps, truncated 的实际语义]

- role_id: wind_or_disturbance_compensation_bonus
  reason: 任务描述提到初始随机力/可能的风力扰动，但 obs 中没有直接暴露风力或扰动强度的观测量，无法构造针对性补偿奖励
  forbidden_or_missing_signals: [obs.wind, obs.disturbance_force]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_distance_shaping | obs/next_obs 的 x_position, y_position | 无 | dense_state_signal, bounded_signal | 建议使用距离差分而非绝对距离，避免尺度问题 |
| velocity_damping_near_target | x_velocity, y_velocity, position | 无 | quadratic_penalty, dense_state_signal | 可与距离做乘积形式，仅在接近目标时生效 |
| orientation_stability | body_angle, angular_velocity | 无 | quadratic_penalty | 角度建议用绝对值或平方惩罚 |
| terminal_outcome_signal | terminated, next_obs 全部字段 | info.success/failure | bounded_signal（终局离散奖励） | 需要用阈值组合近似判断结局类型 |
| contact_bonus | left/right contact, position, velocity | 无 | bounded_signal（一次性触发） | 仅在首次接触且状态合理时触发一次 |
| engine_usage_penalty | action | 无 | bounded_signal（按动作类别赋予小惩罚） | 建议权重远小于主职责 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不着陆 | terminated 很少发生，obs 长期停留在目标上方且速度接近0但contact始终为0 | 增大距离塑形随时间的紧迫性，或对长期悬停施加轻微惩罚 |
| 高速冲撞目标 | crash_or_body_contact 频发，且发生时 obs.y_velocity/x_velocity 绝对值较大 | 加强 velocity_damping_near_target 的权重，尤其在靠近目标时 |
| 姿态失控翻转 | terminated 前 obs.body_angle 或 angular_velocity 持续增大 | 加强 orientation_stability 惩罚，或提前在中等距离引入姿态约束 |
| 越界漂移 | horizontal_position_outside_viewport 频发，且发生前 x_position 持续单向增大 | 检查 goal_distance_shaping 是否对水平方向惩罚不足，适当增加水平方向权重 |
| 引擎滥用/震荡 | action 序列中主引擎和姿态引擎交替频繁触发但位置无明显改善 | 引入 engine_usage_penalty，并检查是否与主职责冲突导致震荡 |
| 触地即视为成功但状态不佳 | contact 变为1时姿态角或速度仍较大，随后仍触发 crash 类终止 | 收紧 contact_bonus 的触发条件阈值，避免过早给予接触奖励 |
