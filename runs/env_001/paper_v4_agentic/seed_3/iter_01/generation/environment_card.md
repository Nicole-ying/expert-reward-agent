# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个 2D 飞行器从初始位置（通常靠近视口顶部中央）出发，尽可能快地降落到场景中央的目标平台上，并以低速度、稳定姿态安全停稳，使两条支撑腿同时接触平台。次要目标：在完成任务的过程中，尽量减少引擎使用量（节省燃料、减少推力）。不应将姿态摆动最小化或单纯的速度最小化作为独立目标，这些只是达成安全着陆的附属约束。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达一个明确的目标位置（中央平台）并实现精确定位/停靠，属于典型的导航目标到达任务。虽然存在推力节省的次要要求，但它不构成与主目标等价的冲突目标，因此不选 multi_objective_task。环境也没有生存平衡、持续前进、抓取操作、强安全约束或极度稀疏探索的特征。动力学子类型进一步细化为目标逼近与软接触（goal_approach_and_soft_contact）。

dynamics_subtype: goal_approach_and_soft_contact

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 通常为 float32（匿名环境未明确，但符合连续观测惯例）
- 各维度含义：
  - obs[0]: x_position（水平坐标，相对于目标平台中心的偏移）—— reward_usable: true
  - obs[1]: y_position（垂直坐标，相对于平台高度基准的偏移）—— reward_usable: true
  - obs[2]: x_velocity（水平线速度）—— reward_usable: true
  - obs[3]: y_velocity（垂直速度）—— reward_usable: true
  - obs[4]: body_angle（机身倾斜角度）—— reward_usable: true
  - obs[5]: angular_velocity（角速度）—— reward_usable: true
  - obs[6]: left_support_contact（左侧支撑腿接触标志，1.0表示接触）—— reward_usable: true
  - obs[7]: right_support_contact（右侧支撑腿接触标志，1.0表示接触）—— reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：
  - action 0: no_engine —— 不启动任何引擎（惯性运动）
  - action 1: left_orientation_engine —— 启动左侧方向引擎（产生逆时针或顺时针力矩，改变姿态）
  - action 2: main_engine —— 启动主引擎（产生向上的推力，通常用于减速或上升）
  - action 3: right_orientation_engine —— 启动右侧方向引擎（产生与左侧引擎相反的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：`body_not_awake_or_settled` 为真，并且可以通过观测信号交叉验证：两条支撑腿均接触平台（obs[6] 和 obs[7] 都为 1.0）、水平位置接近 0（obs[0] ≈ 0）、垂直速度接近 0、姿态角接近水平。这种情况暗示飞行器已稳定停靠在目标平台上。
- **failure-like termination**：`crash_or_body_contact`（主体与地形或其他物体发生不期望的接触，导致损毁）、`horizontal_position_outside_viewport`（水平位置超出屏幕边界，飞行器脱离有效区域）。
- **ambiguous termination**：`body_not_awake_or_settled` 为真，但两条支撑腿未同时接触平台，或者位置不在平台附近。这可能是飞行器在平台外静止但未悬空（例如已经坠毁但引擎关闭或卡在地形中），需通过位置和接触信号判别。在初始学习阶段，部分此类终止可视为失败。
- **truncation**：无显式截断逻辑，`info` 为空，`truncated` 返回 False，即 episode 仅在触发上述终止条件时结束。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 内无任何成功标记）
- explicit_failure_flag_available: false（`info` 内无任何失败标记）
- allowed_info_fields: 无（`info` 返回空字典）
- forbidden_or_uncertain_info_fields: 所有 `info` 字段（因为没有声明任何可用字段，且原环境可能将奖励或终止原因隐藏在 `info` 中，但根据要求不能假设其存在，因此全部禁止使用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- 允许使用：`obs`（8 维观测向量）、`action`、`next_obs`（8 维观测向量）
- 禁止使用：`original_reward`、任何未明确允许的 `info` 字段
- 训练进度 `training_progress` 在任务描述中未被要求用于奖励调度，且无明确允许说明，应当避免使用，除非后续 prompt 明确开放此参数。

## 7. 可用于奖励函数的信号
- **位置类**：`x_position` (obs[0])，`y_position` (obs[1])。可计算与目标平台的水平距离和垂直距离，用于引导接近。
- **速度类**：`x_velocity` (obs[2])，`y_velocity` (obs[3])。可用于惩罚着陆时的冲击速度，或在飞行阶段鼓励平滑性。
- **姿态类**：`body_angle` (obs[4])，`angular_velocity` (obs[5])。用于要求安全着陆时的姿态稳定性（尽量接近水平）。
- **接触类**：`left_support_contact` (obs[6])，`right_support_contact` (obs[7])。两腿同时接触平台是成功着陆的必要条件，可据此构造着陆奖励。
- **动作/引擎类**：`action`。可惩罚引擎使用（no_engine 不惩罚，其余动作惩罚）以鼓励节省燃料。
- **派生推断信号（derived_possible）**：
  - 成功着陆指示器：可从 `body_not_awake_or_settled` 导致 episode 结束，且 `obs[6]` 和 `obs[7]` 均为 1.0、obs[0] 接近 0、obs[3] 接近 0 间接推断。可在奖励函数中结合 next_obs 构造着陆成功奖励，但需谨慎使用，因为无法直接读取终止原因。
  - 失败着陆指示器：可从 episode 结束时 `crash_or_body_contact` 或出界未接触双足推断，但同样无法在奖励计算时直接获取，只能通过观测模式判断（如 `next_obs` 中位置突变、速度极大等）。

## 8. 不确定或不可用的信号
- `info` 中任何字段：完全不可用，因为返回空字典，并且我们被禁止假设其内部结构。
- 显式的“成功”或“失败”标志：不可用。
- 燃料消耗量：未在观测中直接提供，只能通过动作计数间接推断。
- 绝对位置（全局坐标）：只给出相对位置，可用。
- 平台大小或着陆区域半径：未提供，但可通过位置阈值合理假设。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete (4-actions)
morphology:
  body_type: 2d_rigid_body_with_two_support_legs
  actuator_type: main_engine (vertical thrust) + two orientation_engines (torque)
  contact_structure: two contact points on bottom legs
primary_objectives:
  - reach_and_settle_on_central_pad (achieve near-zero position, zero velocity, upright attitude, both legs touching)
secondary_objectives:
  - minimize_engine_use (prefer no_engine actions, penalize unnecessary thrust)
  - minimize_landing_time (implicit via sparse shaping, not explicitly measurable without timer)
main_failure_risks:
  - crashing into ground/obstacles (crash_or_body_contact)
  - drifting outside viewport
  - landing but only one leg touching may be unstable and should be treated as failure
  - staying airborne forever or stable but not on platform (ambiguous termination)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_distance_shaping**  
  purpose: 引导飞行器向目标平台移动，缩小水平距离和垂直距离。  
  why_required: 目标到达是核心任务，无此奖励会导致稀疏奖励，学习极难。  
  usable_signals: `x_position`, `y_position`  
  risks: 若距离权重过大，可能鼓励快速撞击地面而不是受控下降；需要与速度/姿态惩罚结合。

- **role_id: safe_landing_contact**  
  purpose: 奖励两条支撑腿同时接触平台且满足位置、速度、姿态平稳条件。  
  why_required: 明确着陆成功条件，提供终端高奖励信号。  
  usable_signals: `left_support_contact`, `right_support_contact`, `x_position`, `y_position`, `x_velocity`, `y_velocity`, `body_angle`（以及 episode 结束时的 `next_obs`）  
  risks: 只能通过观测交叉验证，可能因 sim-to-real 差异导致伪阳性；必须结合位置阈值避免在远处意外接触被错误奖励。

- **role_id: velocity_penalty_at_landing**  
  purpose: 惩罚着陆瞬间过大的垂直和水平速度，强制实现软着陆。  
  why_required: 没有速度约束的话，agent 可能以高速撞击平台后终止，虽接触但不安全。  
  usable_signals: `y_velocity`, `x_velocity` (在疑似着陆时刻应用)  
  risks: 若整个 episode 都施加严重速度惩罚，会抑制必要的飞行机动；必须仅在着陆检测时施加。

- **role_id: attitude_stabilization**  
  purpose: 要求机身角度接近水平（body_angle ≈ 0），角速度小。  
  why_required: 着陆时姿态倾斜可能导致只有单腿接触或倾覆，必须保持水平。  
  usable_signals: `body_angle`, `angular_velocity`  
  risks: 过度惩罚可能导致 agent 不敢使用方向引擎，反而无法调整姿态，需要与goal平衡。

### 10.2 条件职责 conditional_roles
- **role_id: engine_usage_penalty**  
  condition_to_use: 当 agent 已经接近目标平台但尚未着陆时，进一步鼓励减少推力；或者全程轻微惩罚以节省燃料。  
  usable_signals: `action` (action 0 不罚，1/2/3 进行惩罚)  
  risks: 若全程惩罚过重，agent 可能选择完全不使用引擎导致无法抵达平台。建议在完成目标接近后增加惩罚，或采用分段权重。

- **role_id: approach_path_smoothness**  
  condition_to_use: 如果训练出现剧烈摆动或效率低下，可以引入动作变化惩罚，但初始训练阶段为了避免过度约束，可作为 conditional。  
  usable_signals: `action` 与上一个动作比较。  
  risks: 抑制必要的方向调整，可能造成探索困难。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: explicit_timer_penalty**  
  reason: 环境不提供计时器信号，无法直接测量时间；虽然鼓励快速着陆是隐含目标，但缺乏可用信号，不应强行使用训练进度或 step 计数替代（除非后续明确允许使用 `training_progress`）。  
  forbidden_or_missing_signals: 无时间戳、无剩余步数信息。

- **role_id: survival_bonus**  
  reason: 这是一个有限时域的目标到达任务，存活不是目的，反而可能鼓励悬停拖延。此类奖励会与快速着陆冲突。  
  forbidden_or_missing_signals: 无存活时间信息。

- **role_id: exploration_bonus**  
  reason: 状态空间小，任务结构清晰，不需内部探索奖励。  
  forbidden_or_missing_signals: 无适用性。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_distance_shaping | x_position, y_position | 无 | dense_state_signal (negative L2 distance), bounded_signal (clipping) | 可使用带死区的距离函数，避免在平台上微调时持续负奖励 |
| safe_landing_contact | left_support_contact, right_support_contact, x_position, y_position, x_velocity, y_velocity, body_angle | 无显式着陆成功标志 | terminal_bonus (if all conditions met), logical_and | 必须在疑似 episode 结束时，条件为双接触、|x|<ε, |y_vel|<ε, |angle|<ε |
| velocity_penalty_at_landing | y_velocity, x_velocity, (着陆时机通过接触+位置推断) | 无 | quadratic_penalty, conditional_on_event | 事件检测可能利用 next_obs 中接触标志变化或位置骤变 |
| attitude_stabilization | body_angle, angular_velocity | 无 | bounded_signal, absolute value | 可全程或仅在着陆附近使用，需权衡 |
| engine_usage_penalty | action | 无燃料测量 | stepwise_penalty (per non-zero action) | 动作 0 无罚，其他动作罚常数 |
| approach_path_smoothness | action (与前一步比较) | 无 | action_change_penalty | 初始可不用，出现 jitter 时加入 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停空中不肯下降 | 垂直速度始终≈0，episode 很长后才 terminated，y_position 不减小 | 增加高度下降的奖励（负 y 方向）或对 y>0 持续轻微惩罚 |
| 高速撞击平台 | 着陆观测显示速度很大，接触瞬间终止，但获得高距离奖励 | 加强 landing velocity penalty，或只在双接触+低速时才给 terminal bonus |
| 单腿着陆，翻转失效 | 只有一条腿接触，身体倾斜很大，终致 crash 或出界 | 提高 attitude stabilization 权重，增加双接触对称性要求 |
| 过度使用主引擎悬停，燃料浪费 | 高频使用 action 2，但位置变化小 | 引入更积极的引擎使用惩罚，或在完成靠近后再增加燃料节约权重 |
| 向左右飞出视口 | x_position 持续增大，最终出界终止 | 检查 goal_distance_shaping 的水平距离惩罚是否有效，可能需要加大横向纠正 |
| 学习到的策略倾向于