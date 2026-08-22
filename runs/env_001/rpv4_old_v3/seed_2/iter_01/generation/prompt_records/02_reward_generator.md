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

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
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
```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code

**2. components: dict**，记录 individual reward components。

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
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
这是一个 2D 的飞行器/着陆器轨迹优化任务。智能体从一个靠近顶部中央的位置出发，带有随机初始作用力。  
**主目标**：尽快到达画面中央的目标平台，并安全停靠（soft landing），即要求位置收敛到平台上、速度很低、姿态竖直、支撑脚安全触地。  
**次目标**：在完成任务的前提下，尽量少使用引擎推力（节省燃料），同时保持姿态稳定。  
**不应混淆的目标**：该任务不是纯生存或平衡任务——到达目标平台是刚性的成功条件，能量节省只是辅助约束，不是多目标的平等折衷。

## 2. 任务类型选择
**selected_route_id**: `navigation_goal_reaching`  
**confidence**: high  
**reason**: 任务的核心驱动力是移动到指定目标位置（中心平台）并有条件地稳定在那儿。省燃料、快速等只是附加要求，不构成多目标冲突，因此不属于 multi_objective_task。环境没有强调持续行进、驾驶安全或探索稀疏事件，因此 navigation_goal_reaching 的匹配度最高。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: [8]
- **dtype**: float32 (推断)
- **obs[0]**: `x_position`，水平坐标（相对于目标平台中心），用于衡量横向偏移，奖励可用 ✅
- **obs[1]**: `y_position`，垂直坐标（相对于平台高度），用于衡量高度/下降进程，奖励可用 ✅
- **obs[2]**: `x_velocity`，水平线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[3]**: `y_velocity`，垂直线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[4]**: `body_angle`，机体角度（可能以弧度为单位，0 表示竖直），奖励可用 ✅
- **obs[5]**: `angular_velocity`，角速度，用于稳定姿态，奖励可用 ✅
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0`: `no_engine` – 不激活任何引擎，仅靠惯性飞行。
  - `1`: `left_orientation_engine` – 点火左定向引擎，产生一个使机体逆时针（或某一方向）旋转的力矩。
  - `2`: `main_engine` – 点火主引擎，产生向上的推力（对抗重力并控制下降速度）。
  - `3`: `right_orientation_engine` – 点火右定向引擎，产生与动作 1 相反的旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `MASKED_STEP_SOURCE` 的 `terminated` 计算逻辑：
- **crash_or_body_contact**：躯体发生碰撞或身体支撑点接触地面。此情景可能包括：
  - 成功的软着陆（接触时速度极低、姿态正）。
  - 失败的高速撞击/翻倒等。
- **horizontal_position_outside_viewport**：x 坐标超出视区边界，明确失败。
- **body_not_awake_or_settled**：躯体进入睡眠状态或已稳定（settled）。这可能是：
  - 成功着陆后自然睡眠。
  - 坠毁后物理引擎判定身体安定，属于失败。

**总结终止模式**：
- **success-like termination**：`body_not_awake_or_settled` 在速度接近于 0、位置接近目标平台、接触标志为 1 且角度接近 0 时发生。
- **failure-like termination**：`horizontal_position_outside_viewport`；或因高速、大角度导致的 `crash_or_body_contact`/`body_not_awake_or_settled`。
- **ambiguous termination**：`crash_or_body_contact`/`body_not_awake_or_settled` 需结合 next_obs 判定成败。
- **truncation**：未提及，但有最大 episode 长度（通常存在），应在 reward 中注意时间步惩罚。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无任何显式标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因为没有提供）

> 因此，任何成功/失败的判断必须从 `next_obs` 与 `terminated` 的组合中推断，而不能依赖 `info`。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`
  - `action`
  - `next_obs`
  - `info` 中的字段：无（info 为空，使用它没有任何实际效果）
  - `training_progress`：仅在 prompt 明确允许时使用，此处未明确允许，**建议禁止使用**。
- **禁止使用**：
  - `original_reward`（被屏蔽的官方奖励）
  - 任何未声明的 `info` 字段
  - 在 `obs` 中假设的未列出维度

## 7. 可用于奖励函数的信号
- **position**：`x_position`, `y_position`（相对目标），可计算到目标的欧氏距离或分别控制。
- **velocity**：`x_velocity`, `y_velocity`，可设计近地减速、稳定约束。
- **orientation**：`body_angle`，鼓励竖直。
- **angular**：`angular_velocity`，鼓励旋转抑制。
- **contact**：`left_support_contact`, `right_support_contact`，标志着陆接触已发生，可配合速度用于判定软着陆。
- **action/engine**：`action` 选择，可用于惩罚引擎使用（燃料节省）。
- **other**：`terminated` 标志（作为环境返回，但 `compute_reward` 接口未直接提供，可通过在调用 `compute_reward` 时传入 `done` 标志获取；我们可以在训练循环中以额外参数传递，但接口中只有 `next_obs` 和 `info`；`terminated` 通常可在调用时通过另一个变量获得，若必须在 `compute_reward` 中获取，需要从外部传入——在此假设可以在调用时提供 `done` 标志，否则可通过 `next_obs` 的特征推断终止状态。由于题干未明确禁止，但谨慎起见，直接使用 `done` 需要额外参数，在标准接口中不鼓励，所以这里只强调可从 `next_obs` 和 `info` 推断。但 info 空，所以成功稀疏奖励只能基于 `next_obs` 的状态是否符合成功条件给出，而不依赖外部 `done`。所以我们仍可以将“是否处于终止且状态符合成功条件”作为奖励计算的一部分，但必须在训练循环的每个 step 结束时调用，需要能够判断 `done`，这部分由训练框架负责。我们此处声明信号可用，但不具体到实现细节。）

## 8. 不确定或不可用的信号
- `original_reward`：不可用。
- 任何 `info` 字段：不可用（因为 info 为 `{}`）。
- 明确的成功/失败标志：不存在，不能直接用作奖励源。
- 世界坐标系绝对位置：未知，只有相对位置。
- 引擎推力强度、燃料剩余量：未提供，不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete (action space) acting on continuous dynamics
morphology:
  body_type: 矩形或近似多边形机体，底部有两个支撑脚
  actuator_type: 三个引擎（一个主推力引擎，两个定向旋转引擎）
  contact_structure: 两个底部支撑点（left/right），独立检测接触
primary_objectives:
  - 到达目标平台中心位置 (x≈0, y≈0)
  - 实现安全软着陆（低速度、角度竖直、支撑脚稳定接触）
secondary_objectives:
  - 最小化引擎推力使用（节约燃料）
  - 快速完成任务（尽快）
main_failure_risks:
  - 高速撞击平台导致坠毁
  - 姿态失控旋转
  - 飞出水平边界
  - 长时间无法收敛而耗尽训练步数（truncation）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_progress_guidance`  
  **purpose**: 持续引导智能体靠近目标平台，使之尽快到达目标水平位置和高度。  
  **why_required**: 这是核心目标，没有此引导，agent 可能随机游走，无法稳定达到目标。  
  **usable_signals**: `x_position`, `y_position`, `x_velocity`, `y_velocity`  
  **risks**: 单纯基于位置的奖励可能鼓励 agent 高速冲向目标，忽略减速，导致坠毁。

- **role_id**: `soft_landing_safety`  
  **purpose**: 在接近平台时严格要求低速、竖直姿态，防止坠毁。  
  **why_required**: 任务要求“安全接触”，即着陆时损失小、姿态正；否则即使位置到达也算失败。  
  **usable_signals**: `y_velocity`, `x_velocity`, `body_angle`, `angular_velocity`, `left_support_contact`, `right_support_contact`  
  **risks**: 过于严格的惩罚可能使 agent 不敢下降或长时间悬停不触地。

- **role_id**: `energy_efficiency`  
  **purpose**: 鼓励少用主引擎和定向引擎，降低燃料消耗。  
  **why_required**: 任务描述明确要求“as little engine thrust as possible”。  
  **usable_signals**: `action` (特别是 `main_engine` 和两个定向引擎的选择)  
  **risks**: 权重过高会导致 agent 放弃使用引擎，无法克服重力下降，完不成主任务。

### 10.2 条件职责 conditional_roles
- **role_id**: `horizontal_alignment_and_stabilization`  
  **condition_to_use**: 当智能体 y 坐标已经接近 0（已接近着陆高度）或垂直速度已经很低时，加强水平位置和角度的稳定奖励。  
  **usable_signals**: `x_position`, `x_velocity`, `body_angle`, `angular_velocity`  
  **risks**: 过早强化可能干扰飞行轨迹的灵活性。

- **role_id**: `sparse_mission_success_reward`  
  **condition_to_use**: 仅在成功着陆时刻（基于 next_obs 判断满足位置、接触、速度、角度等所有成功条件）提供一次较大的奖励。  
  **usable_signals**: 全部 next_obs 字段，不可依赖 info。  
  **risks**: 稀疏信号可能难以学习，需要与密集奖励共同使用；若定义不严格，可能将失败态误判为成功。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `survival_balance`  
  **reason**: 任务没有持续的平衡生存危机（如倒立摆、独轮车），主目标不是维持存活，而是到达目标。  
  **forbidden_or_missing_signals**: 不具备必要的“生存时间”或“活过多少步”的显式信号。
- **role_id**: `multi_objective_trade_off`  
  **reason**: 虽然有多项指标，但到达并安全着陆是绝对优先，燃料节省只是次要目标，不需要复杂的帕累托折衷，因此不属于 multi_objective_task 框架中的平等多目标。
- **role_id**: `exploration_bonus`  
  **reason**: 环境状态空间相对简单，目标明确，不需要刻意鼓励状态空间探索；盲目添加可能会噪声干扰学习。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_progress_guidance | x_position, y_position | none | distance_penalty (L2), bounded_reward (per‑dimension), potential‑based shaping | 鼓励同时减小水平与垂直偏差 |
| soft_landing_safety | y_velocity, x_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact | 无 | velocity_penalty (clipped), angular_penalty, contact_conditional (仅在接触时加强速度/角度惩罚) | 需要在下降末段激活 |
| energy_efficiency | action (离散值) | 无 | action_cost (对 action≠0 惩罚，或对1,2,3分别加权) | 可与进度耦合，避免“躺平” |
| horizontal_alignment_and_stabilization | x_position, x_velocity, body_angle, angular_velocity | 无 | quadratic_penalty, conditional_scale (在 y 接近 0 时放大) | 增加着陆横向稳定性 |
| sparse_mission_success_reward | 全部 next_obs | 无 | success_flag (布尔逻辑) → fixed_bonus | 必须严格定义成功门限，避免 reward hacking |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 不动或轻微漂浮，不下降 | 动作全程为 0 或主引擎极少点火；y_position 几乎不变 | 增加 goal_progress_guidance 权重，考虑基于高度的势能奖励，或降低能量惩罚 |
| 高速冲向平台坠毁 | y_velocity 大幅为负，接触时速度大 | 强化 soft_landing_safety 在低高度的速度惩罚，或使用速度上限饱和惩罚 |
| 在目标上空严重摇摆或旋转 | body_angle 或 angular_velocity 震荡 | 加强 orientation_convergence 的惩罚权重，或添加角速度阻尼奖励 |
| 飞出水平边界被截断 | x_position 绝对值快速增加 | 增加横向偏离的惩罚，或引入边界激活的惩罚（如 \( \|x\| \) 过大时大幅惩罚） |
| 成功悬停但迟迟不着陆 | y_position 接近零但一直不接触，持续消耗能量 | 在高度很低时引入“着陆鼓励”附加奖励（contact 瞬间奖励）或降低主引擎奖励，或使能量惩罚与时间成比例增加 |
| 稳定着陆后仍继续引擎点火 | 着陆后接触 = 1，但仍选择动作 1,2,3 | 当着陆接触满足速度条件时，加大 action_cost，并给出成功稀疏奖励后截断，或抑制引擎使用 |



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


```
