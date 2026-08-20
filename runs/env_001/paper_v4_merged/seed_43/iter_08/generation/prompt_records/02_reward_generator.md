# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

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



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---

## 4. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"定位到合适的算子变换。
以数学语义和训练表现证据为准，不要求组件名完全匹配。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |





# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -87.190

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress | 1 | -87.190 | -87.190 | unsolved |
| action_cost + landing_contact_reward + progress_shaping + shaped_progress | 1 | -87.190 | -87.190 | unsolved |
| action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | 2 | -105.530 | -105.530 | unsolved |
| action_cost + angle_hinge + danger_penalty + progress_shaping | 1 | -117.480 | -117.480 | unsolved |
| action_cost + angle_hinge + progress_shaping | 1 | -117.880 | -117.880 | unsolved |
| action_cost + angle_hinge + landing_contact_reward + progress_shaping | 1 | -122.170 | -122.170 | unsolved |

## Previous interventions

- iter 2 (score=-117.480, structure=action_cost + angle_hinge + danger_penalty + progress_shaping): 4. `selected_level`：Level 2 结构变换——基于信号缺口与几乎死亡组件的证据，新增使用未利用观测的危险惩罚组件。 | 5. `selected_intervention`：新增 `danger_penalty` 组件，检测 `abs(nx)>1.2`、`ny<-0.2`、`abs(nangle)>0.8`、或速度幅值 >5.0 等致命状态，每命中步给予 −1.0 惩罚。
- iter 3 (score=-122.170, structure=action_cost + angle_hinge + landing_contact_reward + progress_shaping): selected_level：Level 2 — structural transform，因前轮迭代得分停滞且僵尸组件（danger_penalty active_rate=0%）未实现设计意图，需移除并替换为新职责信号。 | selected_intervention：删除danger_penalty，新增landing_contact_reward组件，基于支撑脚接触和到目标距离的连续bounded factor，以提供着陆指向性奖励。
- iter 4 (score=-87.190, structure=action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress): 4. selected_level: Level 2 — structure change: remove zombie angle_hinge and replace with a landing_speed_gate that scales progress_shaping based on speed when close to target. | 5. selected_intervention: Delete angle_hinge; add `landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_next * max(0.0, 1.0 - dist_next / 0.5))` and multiply progress_shaping by it. This one-component swap leaves action_cost an
- iter 5 (score=-87.190, structure=action_cost + landing_contact_reward + progress_shaping + shaped_progress): 4. selected_level: Level 2 – structural change, because the landing_speed_gate component is active 100% of the time and contributes ~100% signed share as a non‑reward artefact, requiring removal from the component output | 5. selected_intervention: Remove `landing_speed_gate` from the returned components dictionary; keep its computation and multiplication intact (used for shaping) but stop emitting it as a reward term.
- iter 6 (score=-114.350, structure=action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress): 4. `selected_level`：Level 2 — 信号覆盖存在缺失（角度约束），需要添加一个新组件，属于结构变换。 | 5. `selected_intervention`：新增`angle_hinge_penalty`组件，对机身角度的绝对值超过0.3 rad的部分施加线性惩罚，系数0.03，引导飞行器保持水平姿态，避免触地坠毁。
- iter 7 (score=-105.530, structure=action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress): 4. `selected_level`：Level 2 — 结构变换，触发条件：无界→有界（progress_shaping的负分支在坠毁时爆炸，需bounding）。 | 5. `selected_intervention`：仅修改progress_shaping组件，从potential-based无界差分变为基于距离增量的bounded improvement（进步系数0.5，退步系数0.05），以压制退步时的灾难性惩罚。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
