# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是 **2D 飞行器着陆控制**：一个搭载主推进器与姿态推进器的刚体从视口顶部附近出发，必须在最短时间内飞向并稳定停靠在画面中央的目标平台上，且尽可能少地使用引擎推力。  
主要目标：**到达目标位置并安全、稳定地着陆**（接近目标、减小速度、保持直立姿态、双腿接触平台）。  
次要目标：耗费更少的燃料（即减少不必要的引擎动作），并尽可能快地完成着陆。  
不该混淆的目标：纯粹的速度最小化或仅追求时间最短而不考虑着陆稳定性。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 任务的核心需求是让飞行器到达并停留在指定的目标平台上，属于典型的“导航至目标点”问题。速度、燃料等仅为附加优化项，并非相互冲突的独立主目标。

动力学子类型 dynamics_subtype: **goal_approach_and_soft_contact**  
（接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position – 相对于目标平台的水平坐标，reward_usable: true
- obs[1]: y_position – 相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity – 水平线速度，reward_usable: true
- obs[3]: y_velocity – 垂直线速度，reward_usable: true
- obs[4]: body_angle – 机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity – 角速度，reward_usable: true
- obs[6]: left_support_contact – 左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact – 右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine（无任何引擎推力）
- action/action_dim 1: left_orientation_engine（启动左侧姿态引擎，施加旋转冲量）
- action/action_dim 2: main_engine（启动主引擎，产生纵向推力）
- action/action_dim 3: right_orientation_engine（启动右侧姿态引擎，施加反向旋转冲量）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:** `body_not_awake_or_settled`（机体停止运动/休眠，通常意味着已经稳定着陆在目标平台上）。
- **failure-like termination:** `crash_or_body_contact`（发生碰撞或除双腿外其他部位接触地面），`horizontal_position_outside_viewport`（水平位置飞出视口边界）。
- **ambiguous termination:** 无。
- **truncation:** 源码未显式提供截断，但环境可能带有默认的最大回合步数，其信息未在给出的 step 源码中体现，暂视为不存在。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 当前 step 源码返回空字典 `{}`
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（不存在）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- 允许使用：
  - obs（当前观测）
  - action（采取的动作）
  - next_obs（下一时刻观测）
  - info（当前仅为空字典）
  - training_progress（仅当 prompt 明确允许时使用，本环境未要求，故暂不使用）
- 禁止使用：
  - original_reward（官方奖励已被屏蔽）
  - 任何未声明的 info 字段
  - 未声明的 obs 切片/外部状态

## 7. 可用于奖励函数的信号
- **position:** x_position, y_position（均为相对于目标的坐标，接近零时表示已到达）
- **velocity:** x_velocity, y_velocity（用于抑制速度，特别是在着陆阶段）
- **orientation:** body_angle, angular_velocity（用于抑制滚动与倾斜）
- **contact:** left_support_contact, right_support_contact（标志是否已安全落在平台上）
- **action/engine:** action index（可区分是否启动引擎、哪一个引擎，以便进行燃料惩罚）
- **other:** 可从组合状态中提取“着陆成功”的复合特征（位置近零、速度极小、角度极小、双腿均接触）

## 8. 不确定或不可用的信号
- 环境未提供目标平台的明确坐标（obs 给出的是相对坐标，因此可直接使用），无不可用之虞。
- 不可用的信号：回合时间或步数计数器（未在观测中给出），环境终止原因标签（info 中无相关字段），官方奖励值 original_reward。
- 不确定的信号：没有任何 info 字段表示成功或失败，所有成功/失败的判断必须由观测状态推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid vehicle/lander
  actuator_type: discrete thrusters (main, left orientation, right orientation)
  contact_structure: two landing legs (left_support, right_support)
primary_objectives:
  - 到达目标平台并稳定着陆（x,y ≈ 0, 速度 ≈ 0, 角度 ≈ 0, 双腿接触）
secondary_objectives:
  - 减少引擎使用（燃料效率）
  - 尽快到达（时间隐含在轨迹中，但不能直接用时间信号）
main_failure_risks:
  - 机体与地面发生非支撑腿的碰撞（crash）
  - 水平飞出视口范围
  - 着陆时姿态过大或反冲导致无法稳定接触
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: progress_to_goal**  
  purpose: 驱动机体朝向目标平台移动。  
  why_required: 任务核心是到达目标点，位置奖励是最基本的引导信号。  
  usable_signals: [x_position, y_position]（可使用欧氏距离或分轴惩罚）  
  risks: 过于激进的接近奖励可能导致 overshoot 或与姿态惩罚冲突。

- **role_id: successful_settle**  
  purpose: 当机体成功稳定着陆时给予一次高额奖励，强化最终状态。  
  why_required: 终止条件中无显式成功标志，环境最终成功只能通过状态推断，必须奖励这种“满足成功条件”的状态。  
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]  
  risks: 若成功条件定义过严，可能导致奖励稀疏；定义过松可能误奖崩溃状态。

- **role_id: engine_efficiency**  
  purpose: 惩罚不必要的引擎动作，鼓励节能。  
  why_required: 任务明确要求“尽可能少用引擎推力”，且不奖励动作会促使更快、更直接地达到目标。  
  usable_signals: [action]（对非零动作施加惩罚）  
  risks: 惩罚过重可能使飞行器不敢启动引擎，导致无法起飞或快速陷入局部无动作策略。

### 10.2 条件职责 conditional_roles
- **role_id: orientation_penalty**  
  condition_to_use: 当飞行器接近目标位置时（如距离小于一定阈值），加强姿态约束；远离目标时弱化或不生效。  
  usable_signals: [body_angle, angular_velocity, x_position, y_position]  
  risks: 过早施加可能导致飞行器不敢做大幅机动；若仅在接近时生效，可避免阻碍前期导航。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: time_to_goal_penalty**  
  reason: 环境不提供当前步数或时间信号，也无法通过观测直接获得“已用时间”，强行使用会依赖不可靠的外部假设。  
  forbidden_or_missing_signals: [step_count, elapsed_time]

- **role_id: survival_bonus**  
  reason: 该任务为 goal-reaching 而非 survival，生存奖励会鼓励长时间生存而忽略着陆任务，与快速着陆冲突。  
  forbidden_or_missing_signals: 无可用生存度量，且与任务本质相悖。

- **role_id: velocity_smoothing**（全局速度惩罚）  
  reason: 前期需要较高速度才能快速接近目标，全局惩罚速度会与快速着陆冲突。只能在接近目标时使用（见 orientation_penalty），不作为全局职责。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| progress_to_goal | x_position, y_position | — | dense_state_signal, bounded_signal | 可使用距离负数或 −‖(x,y)‖ |
| successful_settle | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | — | sparse_event_reward（组合条件触发器） | 需设定相对严格的阈值才能避免误判 |
| engine_efficiency | action | — | action_penalty | 对 action≠0 给予小幅度负奖励 |
| orientation_penalty | body_angle, angular_velocity, x_position, y_position (用于距离门控) | — | bounded_signal, quadratic_penalty, gated_by_progress | 随着距离减小逐步打开 |
| time_to_goal_penalty | — | step_count, elapsed_time | — | 排除 |
| survival_bonus | — | — | — | 排除 |
| velocity_smoothing (global) | — | — | — | 排除 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器长时间悬停而不接近目标，或仅使用极少量引擎 | 距离目标一直很遥远，动作惩罚占比大但无进展 | 降低动作惩罚权重或增加距离奖励的尺度 |
| 高速撞向地面导致 crash | y 速度很大，且 crash 或水平出界频繁 | 增强 y 速度负奖励，尤其是在低高度时 |
| 着陆后弹跳或翻滚 | 接触标志闪烁，x、y 速度未能快速归零 | 强化 successful_settle 条件中速度阈值的敏感性，或为反弹施加惩罚 |
| 只能单腿着陆，无法稳定 | 仅一侧接触标志为 1，机体倾斜角大 | 增加对不对称接触的惩罚或加强 body_angle 的稳定要求 |
| 朝一侧漂移出界 | x_position 绝对值逐渐增大且无纠正 | 增加水平位置惩罚，或对 x_velocity 施加适度限制 |
| 主引擎长时间开机，燃料消耗严重 | “main_engine”动作频繁出现，但状态无明显改善 | 检查距离奖励是否过弱，导致 agent 反复无效加速 |



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子和少量动力学类型示例，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive: `w * signal`
  - penalty: `-w * abs(error)` 或 `-w * error**2`
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：无界状态值可能支配总奖励；状态值可能被占据/刷分，而不代表任务完成。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - `x / (1 + abs(x))`
  - `1 / (1 + k * abs(error))`
  - `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或 velocity/proximity 类信号容易被刷。
- 风险：threshold 过小会导致反馈饱和或无梯度。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * (1 / (1 + k * abs(posture_error)))`
- 使用条件：前进/接近奖励导致不健康冲刺、翻倒或失稳。
- 风险：gate 太严格会抑制探索；跳跃类任务尤其要轻。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 风险：乘积容易塌缩；单一接触或单一事件不能当成功。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

